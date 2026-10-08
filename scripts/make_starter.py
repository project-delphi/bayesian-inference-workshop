#!/usr/bin/env python
"""Generate workshop/<module>.py starter files from solutions/<module>.py.

Convention: in a solution, any function or method whose body begins (after the
docstring) with a marker comment

    # [mXX step N]

is stubbed in the starter: the signature and docstring are kept verbatim and the body is
replaced by `raise NotImplementedError  # Module XX, Step N`. Everything else (imports,
helper code, data generators) is copied unchanged, so signatures in both packages are
guaranteed identical.

Usage:
    python scripts/make_starter.py            # all solution modules
    python scripts/make_starter.py m03_expfam # one module
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARK = re.compile(r"^(\s*)# \[m(\w+) step (\d+)\]\s*$")
DEF = re.compile(r"^(\s*)(async\s+)?def\s+\w+\s*\(")


def stub(src: str) -> str:
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = MARK.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, mod, step = m.group(1), m.group(2), m.group(3)
        out.append(f"{indent}raise NotImplementedError  # Module {mod.lstrip('0') or '0'}, Step {step}\n")
        i += 1
        # Skip the remainder of this body: lines more indented than the def, or blank.
        while i < n:
            line = lines[i]
            if line.strip() == "":
                # Peek: if the next non-blank line is still inside the body, keep skipping.
                j = i
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n and (len(lines[j]) - len(lines[j].lstrip())) >= len(indent):
                    i = j
                    continue
                break
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent >= len(indent):
                i += 1
            else:
                break
    return "".join(out)


def main(argv: list[str]) -> int:
    sol_dir = ROOT / "solutions"
    ws_dir = ROOT / "workshop"
    names = argv or sorted(p.stem for p in sol_dir.glob("m*.py"))
    for name in names:
        src = (sol_dir / f"{name}.py").read_text()
        header = (
            '"""STARTER FILE. Fill in every function marked `raise NotImplementedError`.\n'
            "The reference implementation lives in solutions/ with identical signatures.\n"
            '"""\n'
        )
        # Insert the header after the module docstring if present, else at top.
        body = stub(src)
        (ws_dir / f"{name}.py").write_text(body if body.startswith('"""') else header + body)
        print("wrote", ws_dir / f"{name}.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
