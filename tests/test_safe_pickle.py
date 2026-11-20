import os
import pickle

import pandas as pd
import pytest

from modelrouter.data import safe_pickle


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned > /tmp/modelrouter-pwned",))


def test_loads_a_plain_dataframe(tmp_path):
    # Object-dtype strings, as in RouterBench; pandas 3's Arrow strings need pyarrow
    # reconstructors that are deliberately not allow-listed.
    df = pd.DataFrame(
        {"prompt": pd.Series(["a", "b"], dtype=object), "score": [0.0, 1.0], "cost": [1e-4, 2e-4]}
    )
    df.columns = pd.Index(df.columns, dtype=object)
    path = tmp_path / "df.pkl"
    df.to_pickle(path)
    loaded = safe_pickle.load_dataframe(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_refuses_globals_outside_the_allow_list():
    payload = pickle.dumps(_Exploit())
    with pytest.raises(safe_pickle.DisallowedGlobalError, match="system"):
        safe_pickle.loads_restricted(payload)
    assert not os.path.exists("/tmp/modelrouter-pwned")


def test_refuses_builtins_eval():
    payload = b"cbuiltins\neval\n(S'1+1'\ntR."
    with pytest.raises(safe_pickle.DisallowedGlobalError, match=r"builtins\.eval"):
        safe_pickle.loads_restricted(payload)


def test_rejects_non_dataframe(tmp_path):
    path = tmp_path / "list.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(TypeError, match="DataFrame"):
        safe_pickle.load_dataframe(path)
