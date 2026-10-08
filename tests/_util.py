"""Shared helpers for the test suites: fixed keys, tolerances, Monte Carlo utilities."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SEED = 20261007


def key(i: int = 0) -> jax.Array:
    """A deterministic PRNG key for test `i`."""
    return jax.random.fold_in(jax.random.key(SEED), i)


def mc_close(estimate, truth, std_err, n_se: float = 4.0, msg: str = ""):
    """Assert |estimate - truth| <= n_se * std_err."""
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    std_err = np.asarray(std_err)
    gap = np.abs(estimate - truth)
    ok = np.all(gap <= n_se * std_err + 1e-12)
    assert ok, f"{msg} |est-truth|={gap} exceeds {n_se} SE={n_se*std_err}"


def finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(x))))
