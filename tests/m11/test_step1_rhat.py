import jax
import jax.numpy as jnp
import numpy as np

from workshop.m11_diagnostics import split_rhat
from tests._util import key


def test_step1_iid_chains_have_rhat_near_one():
    chains = jax.random.normal(key(270), (4, 1000, 3))
    r = split_rhat(chains)
    assert r.shape == (3,)
    assert bool(jnp.all(jnp.abs(r - 1.0) < 0.01))


def test_step1_shifted_chains_are_flagged():
    chains = jax.random.normal(key(271), (4, 1000, 2)) + jnp.array([0.0, 1.0, 2.0, 3.0])[:, None, None] * jnp.array([1.0, 0.0])
    r = split_rhat(chains)
    assert float(r[0]) > 1.5 and abs(float(r[1]) - 1.0) < 0.01


def test_step1_split_detects_within_chain_drift():
    # one chain whose mean drifts: unsplit R-hat would be ~1, split must exceed 1.1
    n = 1000
    drift = jnp.linspace(-2, 2, n)
    chains = jax.random.normal(key(272), (1, n, 1)) + drift[None, :, None]
    assert float(split_rhat(chains)[0]) > 1.1
