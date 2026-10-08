"""Module 0 environment check. Exempt from the starter-must-fail rule: this tests the
machine, not participant code."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np


def test_x64_enabled():
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_versions():
    import scipy, matplotlib  # noqa: F401

    major, minor, *_ = (int(p) for p in jax.__version__.split(".")[:2])
    assert (major, minor) >= (0, 4)
    assert tuple(int(p) for p in np.__version__.split(".")[:1]) >= (2,)


def test_cpu_device_present():
    assert any(d.platform == "cpu" for d in jax.devices())


def test_packages_importable():
    importlib.import_module("workshop")
    importlib.import_module("solutions")
