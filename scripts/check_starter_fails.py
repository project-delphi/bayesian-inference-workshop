#!/usr/bin/env python
"""Assert that every test (outside tests/m00) fails or errors against the starter code.

Usage:
    python scripts/check_starter_fails.py            # all modules
    python scripts/check_starter_fails.py m03 m04    # selected modules

Exit status 0 means the starter is honest: nothing passes before the participant has
written code. Any passing test is printed and the script exits 1.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXEMPT = {"m00"}


def main(argv: list[str]) -> int:
    tests_dir = ROOT / "tests"
    if argv:
        targets = [tests_dir / m for m in argv]
    else:
        targets = sorted(p for p in tests_dir.iterdir() if p.is_dir() and p.name.startswith("m") and p.name not in EXEMPT)
    targets = [t for t in targets if t.name not in EXEMPT]
    if not targets:
        print("No module test directories found.")
        return 0
    cmd = [sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider", "-q", *map(str, targets)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    passed = [line for line in out.stdout.splitlines() if line.startswith("PASSED ")]
    if passed:
        print("These tests PASS against the starter code; they are not testing anything:")
        for line in passed:
            print("  ", line)
        return 1
    n_failed = sum(1 for l in out.stdout.splitlines() if l.startswith(("FAILED ", "ERROR ")))
    print(f"OK: no test passes against the starter ({n_failed} failed or errored, as intended).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
