"""List every global (module.name) a pickle references, without unpickling it.

pickletools.genops only parses opcodes; nothing in the file is executed.  The output is the
allow-list candidate set for `modelrouter.data.safe_pickle`.

Usage: python -I pickle_globals.py FILE.pkl
"""

from __future__ import annotations

import collections
import pickletools
import sys


def main(path: str) -> None:
    counts: collections.Counter[str] = collections.Counter()
    recent_strings: list[str] = []
    memo: dict[int, str | None] = {}
    last: str | None = None  # value of the most recent push, for memo bookkeeping
    with open(path, "rb") as fh:
        for op, arg, _pos in pickletools.genops(fh):
            if op.name == "MEMOIZE":
                memo[len(memo)] = last
                continue
            if op.name in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[int(arg)] = last  # type: ignore[arg-type]
                continue
            if op.name in ("BINGET", "LONG_BINGET", "GET"):
                last = memo.get(int(arg))  # type: ignore[arg-type]
                if last is not None:
                    recent_strings = [*recent_strings[-1:], last]
                continue
            last = None
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                last = str(arg)
                recent_strings = [*recent_strings[-1:], last]
            elif op.name == "GLOBAL":
                counts[str(arg).replace(" ", ".")] += 1
            elif op.name == "STACK_GLOBAL" and len(recent_strings) == 2:
                counts[".".join(recent_strings)] += 1
            elif op.name in ("REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"):
                counts[f"<op {op.name}>"] += 1
    for name, n in sorted(counts.items()):
        print(f"{n:8d}  {name}")


if __name__ == "__main__":
    main(sys.argv[1])
