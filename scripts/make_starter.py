#!/usr/bin/env python
"""Generate workshop/<module>.py starter files from solutions/<module>.py.

Convention: in a solution, any function or method whose body begins (after the
docstring) with a marker comment

    # [mXX step N]

is stubbed in the starter: the signature and docstring are kept verbatim and the body is
replaced by `raise NotImplementedError  # Module XX, Step N`. Everything else (imports,
helper code, data generators) is copied unchanged, so signatures in both packages are
guaranteed identical.

Function extents come from the `ast` module, not from indentation, so a body that
contains a multi-line string or a comment at a shallower indent is still removed
whole. A marker anywhere other than the first line of a function body is an error:
the code after it would otherwise be copied into the starter.

Usage:
    python scripts/make_starter.py            # all solution modules
    python scripts/make_starter.py m03_expfam # one module
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARK = re.compile(r"^(\s*)# \[m(\w+) step (\d+)\]\s*$")


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    first = node.body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


def _body_end(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """End (exclusive, 0-based) of the function's body: its last statement plus any
    comment lines after it that are indented deeper than the def. Blank lines are
    taken only when such a comment follows them."""
    end = node.end_lineno
    i = end
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].lstrip().startswith("#") and len(lines[i]) - len(lines[i].lstrip()) > node.col_offset:
            end = i = i + 1
            continue
        break
    return end


def stub(src: str, name: str = "<source>") -> str:
    lines = src.splitlines(keepends=True)
    cuts: list[tuple[int, int, str]] = []  # (first line, end line exclusive, replacement), 0-based
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = _has_docstring(node)
        code = node.body[1:] if doc else node.body
        if not code:
            continue
        # Only comments, blank lines and (without a docstring) the signature lie between
        # the end of the docstring, or the def line, and the first statement.
        start = node.body[0].end_lineno if doc else node.lineno
        marker = next((i for i in range(start, code[0].lineno - 1) if MARK.match(lines[i])), None)
        if marker is None:
            continue
        indent, mod, step = MARK.match(lines[marker]).groups()
        cuts.append((marker, _body_end(lines, node), f"{indent}raise NotImplementedError  # Module {mod.lstrip('0') or '0'}, Step {step}\n"))
    # A marked function nested inside a stubbed body goes with it.
    kept: list[tuple[int, int, str]] = []
    for c in sorted(cuts):
        if not kept or c[0] >= kept[-1][1]:
            kept.append(c)
    stray = [i + 1 for i, line in enumerate(lines) if MARK.match(line) and not any(a <= i < b for a, b, _ in kept)]
    if stray:
        raise ValueError(f"{name}: step markers not at the start of a function body, lines {stray}")
    out: list[str] = []
    pos = 0
    for a, b, repl in kept:
        out += lines[pos:a]
        out.append(repl)
        pos = b
    out += lines[pos:]
    return "".join(out)


def main(argv: list[str]) -> int:
    sol_dir = ROOT / "solutions"
    ws_dir = ROOT / "workshop"
    names = argv or sorted(p.stem for p in sol_dir.glob("m*.py"))
    for name in names:
        path = sol_dir / f"{name}.py"
        (ws_dir / f"{name}.py").write_text(stub(path.read_text(), str(path.relative_to(ROOT))))
        print("wrote", ws_dir / f"{name}.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
