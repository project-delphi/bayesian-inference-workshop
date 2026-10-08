"""Pick the implementation package for notebooks: workshop/ by default, solutions/ when
WORKSHOP_IMPL=solutions is set. Usage: `from notebooks._impl import impl` then
`m03 = impl("m03_expfam")`."""
import importlib
import os

import jax

jax.config.update("jax_enable_x64", True)

PKG = "solutions" if os.environ.get("WORKSHOP_IMPL") == "solutions" else "workshop"


def impl(module: str):
    return importlib.import_module(f"{PKG}.{module}")
