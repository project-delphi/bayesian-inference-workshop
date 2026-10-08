import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m03_expfam import Bernoulli, Categorical


def test_step1_bernoulli_logprob_via_base_class():
    fam = Bernoulli()
    eta = fam.to_natural(jnp.asarray(0.3))
    for x in (0.0, 1.0):
        np.testing.assert_allclose(fam.log_prob(jnp.asarray(x), eta), scipy.stats.bernoulli(0.3).logpmf(x), rtol=1e-12)


def test_step1_categorical_conversions_roundtrip_and_minimal_dim():
    p = jnp.array([0.31, 0.22, 0.14, 0.12, 0.09, 0.12])
    fam = Categorical(6)
    eta = fam.to_natural(p)
    assert eta.shape == (5,)
    assert fam.dim == 5
    np.testing.assert_allclose(fam.to_standard(eta), p, rtol=1e-12)


def test_step1_categorical_logprob_sums_to_one_and_matches():
    p = jnp.array([0.31, 0.22, 0.14, 0.12, 0.09, 0.12])
    fam = Categorical(6)
    eta = fam.to_natural(p)
    lps = jnp.stack([fam.log_prob(jnp.asarray(k), eta) for k in range(6)])
    np.testing.assert_allclose(jnp.exp(lps), p, rtol=1e-12)
    np.testing.assert_allclose(jnp.exp(lps).sum(), 1.0, rtol=1e-12)
    assert fam.sufficient_stats(jnp.asarray(5)).shape == (5,)
    np.testing.assert_array_equal(fam.sufficient_stats(jnp.asarray(5)), jnp.zeros(5))
