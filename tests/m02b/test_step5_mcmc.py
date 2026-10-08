import jax.numpy as jnp
import numpy as np

from workshop.m02b_foundations import grid_cost, grid_log_posterior, grid_moments, log_joint, random_walk_metropolis
from tests._util import key
from tests.m02b._common import LOGLAM_GRID, MU_GRID, X


def test_step5_metropolis_matches_grid_moments():
    log_post, _ = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    mean, cov = grid_moments(log_post, MU_GRID, LOGLAM_GRID)
    samples, acc = random_walk_metropolis(lambda t: log_joint(t, X), key(200), jnp.array([1.5, 1.0]), 60_000, 0.35)
    assert samples.shape == (60_000, 2)
    assert 0.1 < float(acc) < 0.7
    tail = samples[5_000:]
    sds = np.sqrt(np.diag(cov))
    # allow for autocorrelation: roughly 10x fewer effective samples
    se = sds / np.sqrt(tail.shape[0] / 10)
    np.testing.assert_array_less(np.abs(tail.mean(0) - mean), 4 * se)
    np.testing.assert_allclose(tail.std(0), sds, rtol=0.15)


def test_step5_metropolis_is_deterministic_in_key():
    f = lambda t: log_joint(t, X)
    s1, _ = random_walk_metropolis(f, key(201), jnp.array([1.8, 2.0]), 500, 0.3)
    s2, _ = random_walk_metropolis(f, key(201), jnp.array([1.8, 2.0]), 500, 0.3)
    np.testing.assert_array_equal(s1, s2)


def test_step5_grid_cost():
    assert grid_cost(2, 200) == 40_000
    assert grid_cost(10, 200) == 200**10
    assert grid_cost(1, 7) == 7
