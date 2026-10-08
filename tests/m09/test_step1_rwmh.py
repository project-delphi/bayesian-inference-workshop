import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import GAUSS2D_COV, gaussian_2d_log_prob, random_walk_mh
from tests._util import key


def test_step1_rwmh_moments_and_acceptance():
    xs, acc = random_walk_mh(gaussian_2d_log_prob, key(200), jnp.zeros(2), 40_000, 1.0)
    assert xs.shape == (40_000, 2)
    assert 0.3 < float(acc) < 0.7
    xs = xs[2_000:]
    # Generous tolerances: a random walk has an integrated autocorrelation time of O(10).
    np.testing.assert_allclose(xs.mean(0), 0.0, atol=0.08)
    np.testing.assert_allclose(jnp.cov(xs.T), GAUSS2D_COV, atol=0.15)


def test_step1_rwmh_is_deterministic_and_rejections_repeat_states():
    xs1, _ = random_walk_mh(gaussian_2d_log_prob, key(201), jnp.zeros(2), 500, 2.5)
    xs2, acc = random_walk_mh(gaussian_2d_log_prob, key(201), jnp.zeros(2), 500, 2.5)
    np.testing.assert_array_equal(xs1, xs2)
    repeats = jnp.mean(jnp.all(xs1[1:] == xs1[:-1], axis=1))
    np.testing.assert_allclose(repeats, 1 - acc, atol=0.01)
