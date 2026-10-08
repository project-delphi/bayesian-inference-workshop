import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from workshop.data import torsion_angles
from workshop.m15b_diffusion import (
    gaussian_mixture_score,
    make_score_fn,
    ode_log_likelihood,
    ode_log_likelihood_exact,
    probability_flow_drift,
    train_score_model,
)
from tests._util import key

S = jnp.array([[[0.6, 0.2], [0.2, 0.4]]])
M = jnp.array([[0.3, -0.2]])
W = jnp.array([1.0])


def test_step5_drift_formula():
    from workshop.m15b_diffusion import beta

    x = jnp.array([0.4, -1.0])
    t = jnp.asarray(0.3)
    np.testing.assert_allclose(probability_flow_drift(lambda z, tt: -z, x, t), jnp.zeros(2), atol=1e-12)
    np.testing.assert_allclose(probability_flow_drift(lambda z, tt: jnp.zeros_like(z), x, t), -0.5 * beta(t) * x, rtol=1e-12)


def test_step5_exact_score_recovers_gaussian_log_density():
    exact = lambda x, t: gaussian_mixture_score(x, t, W, M, S)
    xs = jax.random.multivariate_normal(key(1530), M[0], S[0], (200,))
    ref = scipy.stats.multivariate_normal(np.asarray(M[0]), np.asarray(S[0])).logpdf(np.asarray(xs))
    ll = jax.vmap(lambda x: ode_log_likelihood_exact(exact, x, n_steps=200))(xs)
    assert float(np.abs(np.asarray(ll) - ref).mean()) < 0.02
    ll_h = jax.vmap(lambda x, k: ode_log_likelihood(exact, x, k, n_steps=200))(xs, jax.random.split(key(1531), 200))
    assert abs(float((np.asarray(ll_h) - ref).mean())) < 0.1  # Hutchinson is unbiased; the mean error is small


def test_step5_trained_model_beats_single_gaussian():
    X = jnp.asarray(torsion_angles())
    params, _ = train_score_model(key(1532), X, n_steps=6000, batch_size=256, lr=2e-3, hidden=128)
    X_test = jnp.asarray(torsion_angles(seed=1, n=200))
    ll = jax.vmap(lambda x: ode_log_likelihood_exact(make_score_fn(params), x, n_steps=200))(X_test)
    gauss = scipy.stats.multivariate_normal(np.asarray(X.mean(0)), np.cov(np.asarray(X).T)).logpdf(np.asarray(X_test))
    assert bool(jnp.all(jnp.isfinite(ll)))
    assert float(ll.mean()) > float(gauss.mean()) + 0.3, (float(ll.mean()), float(gauss.mean()))
