"""Load a pandas DataFrame pickle without running code from the file.

A pickle can name any importable callable and have the unpickler call it, so opening an
untrusted pickle normally means running untrusted code.  This loader follows the "restricting
globals" pattern from the `pickle` docs: the only globals it resolves are the pandas/numpy
reconstructors a plain DataFrame needs.  Anything else (os.system, builtins.eval,
subprocess.Popen, ...) raises before it can be called.

The allow-list came from a static opcode scan of the RouterBench files
(`scripts/pickle_globals.py`), which reads opcodes and executes nothing.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("pandas._libs.internals", "_unpickle_block"),
        # pandas >= 2.2 pickles the public aliases of the same classes.
        ("pandas", "DataFrame"),
        ("pandas", "Index"),
        ("pandas", "RangeIndex"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    }
)

# numpy 2 moved numpy.core to numpy._core; resolve the new location directly instead of
# importing the deprecated shim.
_MODULE_REMAP = {
    "numpy.core.multiarray": "numpy._core.multiarray",
    "numpy.core.numeric": "numpy._core.numeric",
}


class DisallowedGlobalError(pickle.UnpicklingError):
    """The pickle referenced a global outside the allow-list."""


class _AllowListUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise DisallowedGlobalError(f"refusing to resolve global {module}.{name}")
        return super().find_class(_MODULE_REMAP.get(module, module), name)


def loads_restricted(data: bytes) -> Any:
    """Unpickle `data`, resolving only allow-listed globals."""
    return _AllowListUnpickler(io.BytesIO(data)).load()


def load_dataframe(path: str | Path) -> Any:
    """Load a pickled pandas DataFrame from `path` with the allow-list unpickler."""
    import pandas as pd

    with open(path, "rb") as fh:
        obj = _AllowListUnpickler(fh).load()
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(obj).__name__}")
    return obj
