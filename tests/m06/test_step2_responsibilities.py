import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma

from workshop.data import single_cell_embedding
from workshop.m06_cavi import GMMPrior, VarParams, elbo, update_responsibilities
from tests._util import key


def test_step2_rows_normalised_and_match_formula():
    x, _, means, _ = single_cell_embedding()
    x = jnp.asarray(x)
    alpha = jnp.array([120.0, 80.0, 60.0, 40.0])
    m = jnp.asarray(means)
    s2 = jnp.array([0.01, 0.02, 0.03, 0.04])
    r = update_responsibilities(x, alpha, m, s2, 0.35)
    assert r.shape == (600, 4)
    np.testing.assert_allclose(r.sum(1), 1.0, rtol=1e-12)
    elog = digamma(alpha) - digamma(alpha.sum())
    sq = jnp.sum((x[:, None] - m[None]) ** 2, -1) + 2 * s2[None]
    logits = elog[None] - 0.5 * sq / 0.35**2
    ref = jnp.exp(logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True))
    np.testing.assert_allclose(r, ref, rtol=1e-10, atol=1e-14)


def test_step2_no_underflow_far_from_all_centres():
    x = jnp.array([[50.0, 50.0]])
    alpha = jnp.ones(3)
    m = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    r = update_responsibilities(x, alpha, m, jnp.ones(3) * 0.1, 0.35)
    assert bool(jnp.all(jnp.isfinite(r)))
    np.testing.assert_allclose(r.sum(), 1.0, rtol=1e-12)


def test_step2_update_increases_elbo():
    x, _, means, _ = single_cell_embedding()
    x = jnp.asarray(x)
    r0 = jax.random.dirichlet(key(210), jnp.ones(4), (600,))
    alpha = jnp.array([120.0, 80.0, 60.0, 40.0])
    m = jnp.asarray(means)
    s2 = jnp.array([0.01, 0.02, 0.03, 0.04])
    prior = GMMPrior(1.0, 3.0, 0.35)
    before = elbo(x, VarParams(r0, alpha, m, s2), prior)
    r1 = update_responsibilities(x, alpha, m, s2, prior.sigma)
    after = elbo(x, VarParams(r1, alpha, m, s2), prior)
    assert after > before
