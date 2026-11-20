"""Train/validation/test splits.

`random_split` stratifies by benchmark family, so every family appears in every split in the same
proportion.  `leave_family_out` holds out one whole benchmark: the router never sees a prompt from
it during training, which is the closer match to new production traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelrouter.data.dataset import IntArray, RoutingDataset


@dataclass(frozen=True)
class Split:
    name: str
    train: IntArray
    val: IntArray
    test: IntArray

    def check_disjoint(self) -> None:
        sets = [set(self.train.tolist()), set(self.val.tolist()), set(self.test.tolist())]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise AssertionError(f"split {self.name} overlaps")


def random_split(
    ds: RoutingDataset, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.2
) -> Split:
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for fam in sorted(set(ds.family.tolist())):
        idx = np.flatnonzero(ds.family == fam)
        rng.shuffle(idx)
        n_test = round(len(idx) * test_frac)
        n_val = round(len(idx) * val_frac)
        test.append(idx[:n_test])
        val.append(idx[n_test : n_test + n_val])
        train.append(idx[n_test + n_val :])
    split = Split(
        name=f"random-s{seed}",
        train=np.sort(np.concatenate(train)),
        val=np.sort(np.concatenate(val)),
        test=np.sort(np.concatenate(test)),
    )
    split.check_disjoint()
    return split


def leave_family_out(
    ds: RoutingDataset, family: str, seed: int = 0, val_frac: float = 0.1
) -> Split:
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(ds.family == family)
    if len(test) == 0:
        raise ValueError(f"no rows for family {family!r}")
    rest = np.flatnonzero(ds.family != family)
    rng.shuffle(rest)
    n_val = round(len(rest) * val_frac)
    split = Split(
        name=f"lofo-{family}",
        train=np.sort(rest[n_val:]),
        val=np.sort(rest[:n_val]),
        test=np.sort(test),
    )
    split.check_disjoint()
    return split
