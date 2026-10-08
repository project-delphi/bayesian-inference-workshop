import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import HMCInfo, UPLIFT_DIM, uplift_centered_log_prob, uplift_noncentered_log_prob
from workshop.m11_diagnostics import divergences, ess, run_chains, split_rhat
from tests._util import key


def test_step3_divergence_count():
    e = jnp.array([0.1, 2000.0, jnp.nan, -0.5, jnp.inf, 999.0])
    info = HMCInfo(accept=jnp.zeros(6, bool), accept_prob=jnp.zeros(6), energy_error=e, log_prob=jnp.zeros(6))
    assert int(divergences(info)) == 3
    assert int(divergences(info, threshold=500.0)) == 4


def test_step3_centred_vs_noncentred_funnel():
    q0s = 0.1 * jax.random.normal(key(290), (4, UPLIFT_DIM))
    ch_c, info_c, _ = run_chains(uplift_centered_log_prob, key(291), q0s, 1000, 1000)
    ch_n, info_n, _ = run_chains(uplift_noncentered_log_prob, key(291), q0s, 1000, 1000)
    assert ch_c.shape == (4, 1000, UPLIFT_DIM)
    div_c, div_n = int(divergences(info_c)), int(divergences(info_n))
    assert div_c > div_n
    assert div_c >= 10
    assert div_n <= 5
    # log tau mixes much better in the non-centred parameterisation
    assert float(ess(ch_n)[1]) > 2 * float(ess(ch_c)[1])
    assert bool(jnp.all(split_rhat(ch_n) < 1.05))
