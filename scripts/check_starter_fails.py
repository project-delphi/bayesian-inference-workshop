#!/usr/bin/env python
"""Assert that every test (outside tests/m00) fails or errors against the starter code.

Usage:
    python scripts/check_starter_fails.py            # all modules
    python scripts/check_starter_fails.py m03 m04    # selected modules

Exit status 0 means the starter is honest: pytest ran, collected tests, and every one
failed or errored before the participant has written code. Any passing test is printed
and the script exits 1, as does any test that is skipped or marked xfail on the starter,
or a run that collected nothing or did not complete. tests/m00 is exempt.
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
    skipped = sorted({t.name for t in targets if t.name in EXEMPT})
    targets = [t for t in targets if t.name not in EXEMPT]
    if skipped:
        print(f"Skipping {', '.join(skipped)}: exempt from the starter check (environment tests may pass on the starter).")
    if argv and not targets and skipped:
        return 0
    missing = [t.name for t in targets if not t.is_dir()]
    if missing:
        print(f"No such test directory: {', '.join(missing)}")
        return 1
    if not targets:
        print("No module test directories found.")
        return 1
    cmd = [sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider", "-q", *map(str, targets)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    passed = [line for line in out.stdout.splitlines() if line.startswith(("PASSED ", "SKIPPED ", "XPASS ", "XFAIL "))]
    if passed:
        print("These tests pass, skip or xfail against the starter code; they are not testing anything:")
        for line in passed:
            print("  ", line)
        return 1
    n_failed = sum(1 for l in out.stdout.splitlines() if l.startswith(("FAILED ", "ERROR ")))
    # pytest exits 1 when tests ran and some failed. Anything else (0: all passed,
    # 2: interrupted or collection error, 3: internal error, 4: usage error, 5: nothing
    # collected) means the run did not demonstrate that the starter fails.
    if out.returncode != 1 or n_failed == 0:
        print(f"pytest exited with status {out.returncode} after {n_failed} failures; the starter was not checked.")
        print("\n".join((out.stdout + out.stderr).strip().splitlines()[-15:]))
        return 1
    print(f"OK: no test passes against the starter ({n_failed} failed or errored, as intended).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
