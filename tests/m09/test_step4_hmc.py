import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import GAUSS2D_COV, gaussian_2d_log_prob, hmc, standard_normal_log_prob
from tests._util import key


def test_step4_standard_normal_near_exact_integration():
    samples, info = hmc(standard_normal_log_prob, key(220), jnp.zeros(1), 4_000, 0.05, 20)
    assert samples.shape == (4_000, 1)
    assert float(info.accept.mean()) > 0.995
    assert float(jnp.abs(info.energy_error).max()) < 1e-2
    np.testing.assert_allclose(samples.mean(), 0.0, atol=0.08)
    np.testing.assert_allclose(samples.var(), 1.0, atol=0.1)


def test_step4_correlated_gaussian_moments():
    samples, info = jax.jit(lambda k: hmc(gaussian_2d_log_prob, k, jnp.zeros(2), 8_000, 0.3, 8))(key(221))
    s = samples[500:]
    np.testing.assert_allclose(s.mean(0), 0.0, atol=0.08)
    np.testing.assert_allclose(jnp.cov(s.T), GAUSS2D_COV, atol=0.15)
    assert info.energy_error.shape == (8_000,)
