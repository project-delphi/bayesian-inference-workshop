import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m02b_foundations import log_joint, log_likelihood, log_prior, log_prior_constrained
from tests.m02b._common import X


def test_step1_likelihood_matches_scipy():
    theta = jnp.array([1.7, jnp.log(6.0)])
    ref = scipy.stats.norm(1.7, 1 / np.sqrt(6.0)).logpdf(np.asarray(X)).sum()
    np.testing.assert_allclose(log_likelihood(theta, X), ref, rtol=1e-12)


def test_step1_prior_has_jacobian():
    theta = jnp.array([0.3, 1.1])
    lam = jnp.exp(theta[1])
    ref_constrained = scipy.stats.norm(0, 10).logpdf(0.3) + scipy.stats.gamma(2.0, scale=2.0).logpdf(float(lam))
    np.testing.assert_allclose(log_prior_constrained(theta[0], lam), ref_constrained, rtol=1e-12)
    np.testing.assert_allclose(log_prior(theta) - ref_constrained, theta[1], rtol=1e-12)


def test_step1_joint_is_sum():
    theta = jnp.array([1.9, 2.0])
    np.testing.assert_allclose(log_joint(theta, X), log_prior(theta) + log_likelihood(theta, X), rtol=1e-12)
