import jax.numpy as jnp
import numpy as np

from workshop.data import ab_test_conversions
from workshop.m05_conjugacy import BetaBernoulli, posterior_update


def test_step1_update_matches_beta_counting():
    ab = ab_test_conversions()
    prior = BetaBernoulli.from_beta(1.0, 1.0)
    post = posterior_update(prior, jnp.asarray(ab.a, dtype=float))
    a, b = post.beta_params()
    np.testing.assert_allclose(a, 1.0 + ab.a.sum())
    np.testing.assert_allclose(b, 1.0 + len(ab.a) - ab.a.sum())
    assert float(post.nu) == 2.0 + len(ab.a)


def test_step1_log_normalizer_is_log_beta():
    from scipy.special import betaln

    prior = BetaBernoulli.from_beta(2.5, 7.0)
    np.testing.assert_allclose(prior.log_normalizer(), betaln(2.5, 7.0), rtol=1e-12)
