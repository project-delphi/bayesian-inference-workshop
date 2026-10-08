"""Shared pytest configuration.

Run against the starter package (default):

    pytest tests/m03

Run against the reference implementations:

    pytest tests/m03 --solutions

The flag works by aliasing the top-level package name `workshop` to the `solutions`
package before any test module is collected. Both packages use relative imports, so
every cross-module dependency resolves inside whichever package is active.
"""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser):
    parser.addoption(
        "--solutions",
        action="store_true",
        default=False,
        help="Run the tests against solutions/ instead of workshop/.",
    )


def pytest_configure(config):
    # Only the flag selects solutions/. WORKSHOP_IMPL is the notebooks' switch and is
    # deliberately ignored here, so a shell that exported it for Jupyter cannot make
    # `make test` or check_starter_fails.py silently grade the reference code.
    use_solutions = config.getoption("--solutions")
    if use_solutions:
        import solutions

        # Drop anything already imported under the starter name.
        for name in [m for m in sys.modules if m == "workshop" or m.startswith("workshop.")]:
            del sys.modules[name]
        sys.modules["workshop"] = solutions
        config.workshop_impl = "solutions"
    else:
        config.workshop_impl = "workshop"


def pytest_report_header(config):
    header = f"workshop implementation: {config.workshop_impl}"
    if os.environ.get("WORKSHOP_IMPL") and not config.getoption("--solutions"):
        header += " (WORKSHOP_IMPL is set but ignored by pytest; use --solutions)"
    return header
