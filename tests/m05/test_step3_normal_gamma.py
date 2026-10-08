import jax.numpy as jnp
import numpy as np
import scipy.special
import scipy.integrate
import scipy.stats

from workshop.data import qpcr_log_expression
from workshop.m05_conjugacy import NormalGamma, normal_gamma_log_marginal, normal_gamma_update


def _grid_posterior(prior, xs):
    mus = np.linspace(0.5, 3.0, 401)
    lams = np.linspace(0.5, 40.0, 801)
    M, L = np.meshgrid(mus, lams, indexing="ij")
    log_prior = scipy.stats.norm(prior.m, 1 / np.sqrt(prior.kappa * L)).logpdf(M) + scipy.stats.gamma(prior.alpha, scale=1 / prior.beta).logpdf(L)
    log_lik = np.sum(scipy.stats.norm(M[..., None], 1 / np.sqrt(L[..., None])).logpdf(np.asarray(xs)), axis=-1)
    lp = log_prior + log_lik
    w = np.exp(lp - lp.max())
    w /= w.sum()
    return (w * M).sum(), (w * L).sum()


def test_step3_posterior_moments_match_grid():
    xs, _, _ = qpcr_log_expression()
    prior = NormalGamma(jnp.asarray(0.0), jnp.asarray(0.1), jnp.asarray(2.0), jnp.asarray(0.5))
    post = normal_gamma_update(prior, jnp.asarray(xs))
    e_mu, e_lam = _grid_posterior(prior, xs)
    np.testing.assert_allclose(post.m, e_mu, atol=2e-3)
    np.testing.assert_allclose(post.alpha / post.beta, e_lam, rtol=2e-2)


def test_step3_marginal_likelihood_matches_quadrature():
    xs = np.array([1.2, 1.9, 2.3])
    m0, k0, a0, b0 = 1.0, 1.0, 2.0, 1.0
    prior = NormalGamma(jnp.asarray(m0), jnp.asarray(k0), jnp.asarray(a0), jnp.asarray(b0))
    log_norm_gamma_const = a0 * np.log(b0) - scipy.special.gammaln(a0)

    def inner(lam):
        # integrate the Gaussian-in-mu part on a fine grid (vectorised)
        mus = np.linspace(-4, 7, 4001)
        log_p = (
            0.5 * np.log(k0 * lam / (2 * np.pi))
            - 0.5 * k0 * lam * (mus - m0) ** 2
            + np.sum(0.5 * np.log(lam / (2 * np.pi)) - 0.5 * lam * (xs[:, None] - mus[None, :]) ** 2, axis=0)
        )
        return np.trapezoid(np.exp(log_p), mus)

    def integrand(lam):
        log_gamma_pdf = log_norm_gamma_const + (a0 - 1) * np.log(lam) - b0 * lam
        return np.exp(log_gamma_pdf) * inner(lam)

    val, _ = scipy.integrate.quad(integrand, 1e-8, 60, limit=200)
    np.testing.assert_allclose(normal_gamma_log_marginal(prior, jnp.asarray(xs)), np.log(val), rtol=1e-5)
