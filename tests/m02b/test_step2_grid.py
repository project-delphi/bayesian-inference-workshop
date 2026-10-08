import jax.numpy as jnp
import numpy as np
import scipy.integrate

from workshop.m02b_foundations import grid_log_posterior, grid_moments, log_joint
from tests.m02b._common import LOGLAM_GRID, MU_GRID, X, np_log_joint_constrained


def test_step2_normalised_and_evidence_matches_quadrature():
    log_post, log_ev = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    assert log_post.shape == (MU_GRID.shape[0], LOGLAM_GRID.shape[0])
    area = float((MU_GRID[1] - MU_GRID[0]) * (LOGLAM_GRID[1] - LOGLAM_GRID[0]))
    np.testing.assert_allclose(jnp.sum(jnp.exp(log_post)) * area, 1.0, rtol=1e-10)
    shift = float(np_log_joint_constrained(1.8, 9.0))
    val, _ = scipy.integrate.dblquad(lambda lam, mu: np.exp(np_log_joint_constrained(mu, lam) - shift), 1.0, 2.6, 0.5, 60.0)
    np.testing.assert_allclose(log_ev, np.log(val) + shift, atol=2e-3)


def test_step2_moments():
    log_post, _ = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    mean, cov = grid_moments(log_post, MU_GRID, LOGLAM_GRID)
    assert mean.shape == (2,) and cov.shape == (2, 2)
    # posterior mean of mu is close to the sample mean (prior is very weak)
    np.testing.assert_allclose(mean[0], float(X.mean()), atol=5e-3)
    assert 0.05 < float(jnp.sqrt(cov[0, 0])) < 0.12
    assert float(jnp.sqrt(cov[1, 1])) > 0.3
    np.testing.assert_allclose(cov[0, 1], cov[1, 0], rtol=1e-8)
    assert bool(jnp.all(jnp.linalg.eigvalsh(cov) > 0))
