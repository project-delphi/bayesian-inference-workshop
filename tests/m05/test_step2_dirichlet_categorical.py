import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from workshop.data import channel_attribution
from workshop.m05_conjugacy import DirichletCategorical, posterior_update


def test_step2_alpha_roundtrip_and_update():
    labels, p_true = channel_attribution()
    alpha0 = jnp.ones(6) * 0.5
    prior = DirichletCategorical.from_alpha(alpha0)
    np.testing.assert_allclose(prior.alpha(), alpha0)
    post = posterior_update(prior, jnp.asarray(labels))
    counts = np.bincount(labels, minlength=6)
    np.testing.assert_allclose(post.alpha(), alpha0 + counts)
    post_mean = post.alpha() / post.alpha().sum()
    assert np.abs(post_mean - p_true).max() < 0.03


def test_step2_log_normalizer():
    alpha = jnp.array([2.0, 0.7, 5.0, 1.3])
    prior = DirichletCategorical.from_alpha(alpha)
    ref = gammaln(np.asarray(alpha)).sum() - gammaln(np.asarray(alpha).sum())
    np.testing.assert_allclose(prior.log_normalizer(), ref, rtol=1e-12)
