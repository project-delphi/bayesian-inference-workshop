import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m01_jax import batched_gaussian_logpdf, gaussian_logpdf
from tests._util import key


def _cov(k, d):
    A = jax.random.normal(k, (d, d))
    return A @ A.T + d * jnp.eye(d)


def test_step2_single_matches_scipy():
    d = 4
    cov = _cov(key(3), d)
    mu = jnp.arange(d, dtype=float)
    x = jax.random.normal(key(4), (d,))
    ref = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(cov)).logpdf(np.asarray(x))
    np.testing.assert_allclose(gaussian_logpdf(x, mu, cov), ref, rtol=1e-10)


def test_step2_batched_matches_scipy():
    d, n = 3, 25
    cov = _cov(key(5), d)
    mu = jnp.zeros(d)
    xs = jax.random.normal(key(6), (n, d))
    ref = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(cov)).logpdf(np.asarray(xs))
    out = batched_gaussian_logpdf(xs, mu, cov)
    assert out.shape == (n,)
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_step2_jit_and_grad_work():
    d = 3
    cov = _cov(key(7), d)
    mu = jnp.zeros(d)
    x = jnp.ones(d)
    g = jax.jit(jax.grad(gaussian_logpdf))(x, mu, cov)
    # d/dx log N(x; mu, cov) = -cov^{-1}(x - mu)
    np.testing.assert_allclose(g, -jnp.linalg.solve(cov, x - mu), rtol=1e-10)
