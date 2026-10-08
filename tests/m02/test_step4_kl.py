import jax
import jax.numpy as jnp
import numpy as np

from workshop.m02_diagnostic import kl_gaussians
from tests._util import key, mc_close


def _spd(k, d):
    A = jax.random.normal(k, (d, d))
    return A @ A.T + d * jnp.eye(d)


def test_step4_zero_for_identical():
    cov = _spd(key(50), 3)
    mu = jnp.ones(3)
    np.testing.assert_allclose(kl_gaussians(mu, cov, mu, cov), 0.0, atol=1e-10)


def test_step4_univariate_formula():
    m0, s0, m1, s1 = 0.5, 1.2, -0.3, 0.8
    ref = np.log(s1 / s0) + (s0**2 + (m0 - m1) ** 2) / (2 * s1**2) - 0.5
    out = kl_gaussians(jnp.array([m0]), jnp.array([[s0**2]]), jnp.array([m1]), jnp.array([[s1**2]]))
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_step4_matches_monte_carlo():
    d = 3
    cov0, cov1 = _spd(key(51), d), _spd(key(52), d)
    mu0, mu1 = jnp.zeros(d), 0.5 * jnp.ones(d)
    xs = jax.random.multivariate_normal(key(53), mu0, cov0, (200_000,))
    lp0 = jax.scipy.stats.multivariate_normal.logpdf(xs, mu0, cov0)
    lp1 = jax.scipy.stats.multivariate_normal.logpdf(xs, mu1, cov1)
    diff = lp0 - lp1
    mc_close(kl_gaussians(mu0, cov0, mu1, cov1), diff.mean(), diff.std() / np.sqrt(xs.shape[0]), msg="KL")
