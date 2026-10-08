import jax.numpy as jnp
import numpy as np
import scipy.special

from workshop.m01_jax import logsumexp
from tests._util import key
import jax


def test_step1_matches_scipy_1d():
    a = jax.random.normal(key(1), (50,)) * 3
    np.testing.assert_allclose(logsumexp(a), scipy.special.logsumexp(np.asarray(a)), rtol=1e-12)


def test_step1_axis_reduction():
    a = jax.random.normal(key(2), (6, 7))
    np.testing.assert_allclose(logsumexp(a, axis=1), scipy.special.logsumexp(np.asarray(a), axis=1), rtol=1e-12)
    np.testing.assert_allclose(logsumexp(a, axis=0), scipy.special.logsumexp(np.asarray(a), axis=0), rtol=1e-12)
    assert logsumexp(a, axis=1).shape == (6,)


def test_step1_no_overflow():
    a = jnp.array([1e4, 1e4 - 1.0])
    out = logsumexp(a)
    assert jnp.isfinite(out)
    np.testing.assert_allclose(out, 1e4 + np.log1p(np.exp(-1.0)), rtol=1e-12)


def test_step1_all_neg_inf_gives_neg_inf_not_nan():
    a = jnp.array([-jnp.inf, -jnp.inf])
    assert logsumexp(a) == -jnp.inf
    assert not jnp.isnan(logsumexp(a))
