import jax.numpy as jnp
import numpy as np
import scipy.integrate
import scipy.stats

from workshop.m05_conjugacy import BetaBernoulli, DirichletCategorical, log_marginal_likelihood, posterior_predictive_logpdf, posterior_update


def test_step4_beta_bernoulli_evidence_vs_quadrature():
    xs = jnp.array([1.0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    a, b = 2.0, 5.0
    prior = BetaBernoulli.from_beta(a, b)
    k, n = float(xs.sum()), xs.shape[0]
    val, _ = scipy.integrate.quad(lambda p: scipy.stats.beta(a, b).pdf(p) * p**k * (1 - p) ** (n - k), 0, 1)
    np.testing.assert_allclose(log_marginal_likelihood(prior, xs), np.log(val), rtol=1e-8)


def test_step4_posterior_predictive_is_posterior_mean():
    xs = jnp.array([1.0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    post = posterior_update(BetaBernoulli.from_beta(2.0, 5.0), xs)
    a, b = post.beta_params()
    np.testing.assert_allclose(jnp.exp(posterior_predictive_logpdf(post, jnp.asarray(1.0))), a / (a + b), rtol=1e-10)
    np.testing.assert_allclose(jnp.exp(posterior_predictive_logpdf(post, jnp.asarray(0.0))), b / (a + b), rtol=1e-10)


def test_step4_dirichlet_predictive_sums_to_one_and_evidence_decomposes():
    labels = jnp.array([0, 2, 2, 1, 3, 2, 0, 0])
    prior = DirichletCategorical.from_alpha(jnp.array([1.0, 1.0, 1.0, 1.0]))
    post = posterior_update(prior, labels)
    probs = jnp.exp(jnp.stack([posterior_predictive_logpdf(post, jnp.asarray(k)) for k in range(4)]))
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-10)
    # chain rule: evidence = sum of sequential predictives
    seq = 0.0
    cur = prior
    for x in labels:
        seq = seq + posterior_predictive_logpdf(cur, x)
        cur = posterior_update(cur, x[None])
    np.testing.assert_allclose(log_marginal_likelihood(prior, labels), seq, rtol=1e-10)
