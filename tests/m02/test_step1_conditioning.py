import jax
import jax.numpy as jnp
import numpy as np

from workshop.m02_diagnostic import gaussian_condition
from tests._util import key


def _spd(k, d):
    A = jax.random.normal(k, (d, d))
    return A @ A.T + d * jnp.eye(d)


def test_step1_matches_brute_force_formula():
    d = 5
    cov = _spd(key(30), d)
    mu = jnp.arange(d, dtype=float)
    obs = jnp.array([1, 3])
    free = jnp.array([0, 2, 4])
    x_obs = jnp.array([0.5, -2.0])
    mu_c, cov_c = gaussian_condition(mu, cov, obs, x_obs)
    S_oo_inv = jnp.linalg.inv(cov[jnp.ix_(obs, obs)])
    ref_mu = mu[free] + cov[jnp.ix_(free, obs)] @ S_oo_inv @ (x_obs - mu[obs])
    ref_cov = cov[jnp.ix_(free, free)] - cov[jnp.ix_(free, obs)] @ S_oo_inv @ cov[jnp.ix_(obs, free)]
    np.testing.assert_allclose(mu_c, ref_mu, rtol=1e-10)
    np.testing.assert_allclose(cov_c, ref_cov, rtol=1e-10)


def test_step1_conditioning_reduces_variance_and_is_consistent_with_sampling():
    d = 3
    cov = _spd(key(31), d)
    mu = jnp.zeros(d)
    obs = jnp.array([2])
    xs = jax.random.multivariate_normal(key(32), mu, cov, (400_000,))
    sel = jnp.abs(xs[:, 2] - 1.0) < 0.02
    emp_mean = xs[sel][:, :2].mean(0)
    mu_c, cov_c = gaussian_condition(mu, cov, obs, jnp.array([1.0]))
    assert bool(jnp.all(jnp.diag(cov_c) < jnp.diag(cov)[:2]))
    np.testing.assert_allclose(emp_mean, mu_c, atol=0.05)
