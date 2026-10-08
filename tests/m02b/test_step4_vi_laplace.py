import jax.numpy as jnp
import numpy as np

import jax

from workshop.m02b_foundations import elbo_gaussian, fit_gaussian_vi, gaussian_kl_to_grid, grid_log_posterior, grid_moments, laplace_approx, log_joint
from tests._util import key
from tests.m02b._common import LOGLAM_GRID, MU_GRID, X


def _grid():
    log_post, _ = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    return log_post, grid_moments(log_post, MU_GRID, LOGLAM_GRID)


def test_step4_kl_nonnegative_and_zero_only_near_truth():
    log_post, (mean, cov) = _grid()
    sds = jnp.sqrt(jnp.diag(cov))
    kl_good = gaussian_kl_to_grid(mean, jnp.log(sds), log_post, MU_GRID, LOGLAM_GRID)
    kl_bad = gaussian_kl_to_grid(mean + jnp.array([0.2, 0.5]), jnp.log(sds), log_post, MU_GRID, LOGLAM_GRID)
    assert kl_good > -1e-6
    assert kl_bad > kl_good + 0.5


def test_step4_elbo_is_a_lower_bound_on_log_evidence():
    log_post, (mean, cov) = _grid()
    _, log_ev = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    eps = jax.random.normal(key(210), (2000, 2))
    sds = jnp.sqrt(jnp.diag(cov))
    elbo = elbo_gaussian(mean, jnp.log(sds), eps, log_joint, X)
    assert float(elbo) <= float(log_ev) + 0.02
    assert float(log_ev) - float(elbo) < 0.2  # a Gaussian is a decent fit here


def test_step4_fit_vi_recovers_mean_and_is_underdispersed():
    log_post, (mean, cov) = _grid()
    eps = jax.random.normal(key(211), (256, 2))
    q_mean, q_log_sd, trace = fit_gaussian_vi(log_joint, X, eps, jnp.array([1.5, 1.0]), jnp.log(jnp.array([0.3, 0.6])))
    assert trace.shape[0] == 500
    assert float(trace[-1]) > float(trace[0])
    kl_final = gaussian_kl_to_grid(q_mean, q_log_sd, log_post, MU_GRID, LOGLAM_GRID)
    assert 0.0 <= float(kl_final) < 0.1
    np.testing.assert_allclose(q_mean, mean, atol=0.05)
    assert bool(jnp.all(jnp.exp(q_log_sd) <= jnp.sqrt(jnp.diag(cov)) * 1.05))


def test_step4_laplace_mode_and_covariance():
    log_post, (mean, cov) = _grid()
    mode, lcov = laplace_approx(log_joint, X, jnp.array([1.5, 1.0]))
    np.testing.assert_allclose(mode[0], mean[0], atol=0.02)
    assert abs(float(mode[1]) - float(mean[1])) < 0.3  # mode of log-precision sits below its mean
    np.testing.assert_allclose(jnp.sqrt(lcov[0, 0]), jnp.sqrt(cov[0, 0]), rtol=0.15)
    assert bool(jnp.all(jnp.linalg.eigvalsh(lcov) > 0))
