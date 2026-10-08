"""Module 5 — Conjugacy as natural-parameter arithmetic.

If x_i ~ p(x | eta) = h(x) exp(eta . t(x) - A(eta)) and the prior on eta is

    p(eta | chi, nu) = exp( chi . eta - nu A(eta) - B(chi, nu) ),

then the posterior after n observations is the same family with

    chi' = chi + sum_i t(x_i),      nu' = nu + n.

The marginal likelihood is a ratio of prior normalisers,
log p(x_1:n) = B(chi', nu') - B(chi, nu) + sum_i log h(x_i), and the posterior predictive
is the one-step version of the same identity. Only B is family-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

from .m03_expfam import Bernoulli, Categorical, ExpFam

Array = jax.Array


@dataclass(frozen=True)
class ConjugatePrior:
    """Conjugate prior over the natural parameter of `fam`, with parameters (chi, nu)."""

    fam: ExpFam
    chi: Array
    nu: Array

    def log_normalizer(self) -> Array:
        """B(chi, nu) = log integral exp(chi . eta - nu A(eta)) d eta. Family specific."""
        raise NotImplementedError

    def with_params(self, chi: Array, nu: Array) -> "ConjugatePrior":
        return replace(self, chi=chi, nu=nu)


# --------------------------------------------------------------------------------------
# Step 1: the generic update
# --------------------------------------------------------------------------------------
def posterior_update(prior: ConjugatePrior, xs: Array) -> ConjugatePrior:
    """Return the posterior after observing xs (first axis indexes observations).

    chi' = chi + sum_i t(x_i), nu' = nu + n. Use jax.vmap over fam.sufficient_stats."""
    # [m05 step 1]
    T = jax.vmap(prior.fam.sufficient_stats)(xs)
    return prior.with_params(prior.chi + T.sum(0), prior.nu + xs.shape[0])


class BetaBernoulli(ConjugatePrior):
    """Beta(a, b) prior on p, expressed over eta = logit p: chi = a, nu = a + b.

    (The Jacobian of p -> eta turns p^{a-1}(1-p)^{b-1} into exp(a eta - (a+b) A(eta)).)
    """

    @staticmethod
    def from_beta(a: float, b: float) -> "BetaBernoulli":
        return BetaBernoulli(Bernoulli(), jnp.array([float(a)]), jnp.asarray(float(a + b)))

    def beta_params(self) -> tuple[Array, Array]:
        """Return (a, b) of the equivalent Beta distribution on p."""
        # [m05 step 1]
        a = self.chi[0]
        return a, self.nu - a

    def log_normalizer(self) -> Array:
        """B(chi, nu) = log Beta(a, b) with (a, b) = beta_params().

        Compute it as lgamma(a) + lgamma(b) - lgamma(a + b); jax.scipy.special.betaln
        is only accurate to about 1e-7 in float64."""
        # [m05 step 1]
        a, b = self.beta_params()
        return gammaln(a) + gammaln(b) - gammaln(a + b)


# --------------------------------------------------------------------------------------
# Step 2: Dirichlet-Categorical
# --------------------------------------------------------------------------------------
class DirichletCategorical(ConjugatePrior):
    """Dir(alpha) prior on p (length K), expressed over the minimal natural parameter of
    Categorical(K): chi = alpha[:-1], nu = sum(alpha)."""

    @staticmethod
    def from_alpha(alpha: Array) -> "DirichletCategorical":
        alpha = jnp.asarray(alpha, dtype=float)
        return DirichletCategorical(Categorical(alpha.shape[0]), alpha[:-1], alpha.sum())

    def alpha(self) -> Array:
        """Recover the full length-K concentration vector."""
        # [m05 step 2]
        return jnp.concatenate([self.chi, jnp.reshape(self.nu - self.chi.sum(), (1,))])

    def log_normalizer(self) -> Array:
        """Log multivariate Beta function of alpha()."""
        # [m05 step 2]
        a = self.alpha()
        return jnp.sum(gammaln(a)) - gammaln(a.sum())


# --------------------------------------------------------------------------------------
# Step 3: Normal-Gamma for a Gaussian with unknown mean and precision
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalGamma:
    """mu | lam ~ N(m, (kappa lam)^{-1}),  lam ~ Gamma(alpha, beta)."""

    m: Array
    kappa: Array
    alpha: Array
    beta: Array


def normal_gamma_update(prior: NormalGamma, xs: Array) -> NormalGamma:
    """Closed-form posterior for n iid N(mu, lam^{-1}) observations."""
    # [m05 step 3]
    n = xs.shape[0]
    xbar = xs.mean()
    ss = jnp.sum((xs - xbar) ** 2)
    kappa_n = prior.kappa + n
    m_n = (prior.kappa * prior.m + n * xbar) / kappa_n
    alpha_n = prior.alpha + 0.5 * n
    beta_n = prior.beta + 0.5 * ss + 0.5 * prior.kappa * n * (xbar - prior.m) ** 2 / kappa_n
    return NormalGamma(m_n, kappa_n, alpha_n, beta_n)


def normal_gamma_log_normalizer(p: NormalGamma) -> Array:
    """log of the normalising constant of the (unnormalised) Normal-Gamma density
    lam^{alpha - 1/2} exp(-beta lam) exp(-kappa lam (mu - m)^2 / 2):
        lgamma(alpha) - alpha log beta + 1/2 log(2 pi / kappa)."""
    # [m05 step 3]
    return gammaln(p.alpha) - p.alpha * jnp.log(p.beta) + 0.5 * jnp.log(2 * jnp.pi / p.kappa)


def normal_gamma_log_marginal(prior: NormalGamma, xs: Array) -> Array:
    """log p(x_1:n) = log Z_post - log Z_prior - n/2 log(2 pi)."""
    # [m05 step 3]
    post = normal_gamma_update(prior, xs)
    n = xs.shape[0]
    return normal_gamma_log_normalizer(post) - normal_gamma_log_normalizer(prior) - 0.5 * n * jnp.log(2 * jnp.pi)


# --------------------------------------------------------------------------------------
# Step 4: evidence and prediction from the normaliser
# --------------------------------------------------------------------------------------
def log_marginal_likelihood(prior: ConjugatePrior, xs: Array) -> Array:
    """log p(x_1:n) = B(chi', nu') - B(chi, nu) + sum_i log h(x_i)."""
    # [m05 step 4]
    post = posterior_update(prior, xs)
    log_h = jax.vmap(prior.fam.log_base_measure)(xs).sum()
    return post.log_normalizer() - prior.log_normalizer() + log_h


def posterior_predictive_logpdf(post: ConjugatePrior, x: Array) -> Array:
    """log p(x_new | x_1:n) = B(chi' + t(x), nu' + 1) - B(chi', nu') + log h(x)."""
    # [m05 step 4]
    one = post.with_params(post.chi + post.fam.sufficient_stats(x), post.nu + 1)
    return one.log_normalizer() - post.log_normalizer() + post.fam.log_base_measure(x)


# --------------------------------------------------------------------------------------
# Step 5: streaming updates and a decision
# --------------------------------------------------------------------------------------
def sequential_update(prior: ConjugatePrior, xs: Array) -> tuple[ConjugatePrior, Array, Array]:
    """Process xs one at a time with lax.scan.

    Returns (final posterior, chi_trajectory [n, dim], nu_trajectory [n])."""
    # [m05 step 5]

    def body(carry, x):
        chi, nu = carry
        chi = chi + prior.fam.sufficient_stats(x)
        nu = nu + 1.0
        return (chi, nu), (chi, nu)

    (chi, nu), (chis, nus) = lax.scan(body, (prior.chi, prior.nu), xs)
    return prior.with_params(chi, nu), chis, nus


def prob_b_beats_a(post_a: BetaBernoulli, post_b: BetaBernoulli, key: Array, n: int) -> Array:
    """P(p_B > p_A | data) by Monte Carlo from the two Beta posteriors."""
    # [m05 step 5]
    ka, kb = jax.random.split(key)
    aa, ba = post_a.beta_params()
    ab, bb = post_b.beta_params()
    pa = jax.random.beta(ka, aa, ba, (n,))
    pb = jax.random.beta(kb, ab, bb, (n,))
    return jnp.mean(pb > pa)
