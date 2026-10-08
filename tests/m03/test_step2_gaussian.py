import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m03_expfam import Gaussian
from tests._util import key


def _spd(k, d):
    A = jax.random.normal(k, (d, d))
    return A @ A.T + d * jnp.eye(d)


def test_step2_roundtrip():
    d = 3
    cov = _spd(key(60), d)
    mu = jnp.array([1.0, -2.0, 0.5])
    fam = Gaussian(d)
    eta = fam.to_natural(mu, cov)
    assert eta.shape == (d + d * d,)
    mu2, cov2 = fam.to_standard(eta)
    np.testing.assert_allclose(mu2, mu, rtol=1e-10)
    np.testing.assert_allclose(cov2, cov, rtol=1e-10)


def test_step2_logprob_matches_scipy():
    d = 3
    cov = _spd(key(61), d)
    mu = jnp.array([1.0, -2.0, 0.5])
    fam = Gaussian(d)
    eta = fam.to_natural(mu, cov)
    xs = jax.random.normal(key(62), (10, d))
    out = jax.vmap(fam.log_prob, in_axes=(0, None))(xs, eta)
    ref = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(cov)).logpdf(np.asarray(xs))
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_step2_univariate_special_case():
    fam = Gaussian(1)
    eta = fam.to_natural(jnp.array([0.7]), jnp.array([[2.5]]))
    np.testing.assert_allclose(fam.log_prob(jnp.array([-0.4]), eta), scipy.stats.norm(0.7, np.sqrt(2.5)).logpdf(-0.4), rtol=1e-12)
