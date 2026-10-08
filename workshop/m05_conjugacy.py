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
    raise NotImplementedError  # Module 5, Step 1


class BetaBernoulli(ConjugatePrior):
    """Beta(a, b) prior on p, expressed over eta = logit p: chi = a, nu = a + b.

    (The Jacobian of p -> eta turns p^{a-1}(1-p)^{b-1} into exp(a eta - (a+b) A(eta)).)
    """

    @staticmethod
    def from_beta(a: float, b: float) -> "BetaBernoulli":
        return BetaBernoulli(Bernoulli(), jnp.array([float(a)]), jnp.asarray(float(a + b)))

    def beta_params(self) -> tuple[Array, Array]:
        """Return (a, b) of the equivalent Beta distribution on p."""
        raise NotImplementedError  # Module 5, Step 1

    def log_normalizer(self) -> Array:
        """B(chi, nu) = log Beta(a, b) with (a, b) = beta_params().

        Compute it as lgamma(a) + lgamma(b) - lgamma(a + b); jax.scipy.special.betaln
        is only accurate to about 1e-7 in float64."""
        raise NotImplementedError  # Module 5, Step 1


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
        raise NotImplementedError  # Module 5, Step 2

    def log_normalizer(self) -> Array:
        """Log multivariate Beta function of alpha()."""
        raise NotImplementedError  # Module 5, Step 2


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
    raise NotImplementedError  # Module 5, Step 3


def normal_gamma_log_normalizer(p: NormalGamma) -> Array:
    """log of the normalising constant of the (unnormalised) Normal-Gamma density
    lam^{alpha - 1/2} exp(-beta lam) exp(-kappa lam (mu - m)^2 / 2):
        lgamma(alpha) - alpha log beta + 1/2 log(2 pi / kappa)."""
    raise NotImplementedError  # Module 5, Step 3


def normal_gamma_log_marginal(prior: NormalGamma, xs: Array) -> Array:
    """log p(x_1:n) = log Z_post - log Z_prior - n/2 log(2 pi)."""
    raise NotImplementedError  # Module 5, Step 3


# --------------------------------------------------------------------------------------
# Step 4: evidence and prediction from the normaliser
# --------------------------------------------------------------------------------------
def log_marginal_likelihood(prior: ConjugatePrior, xs: Array) -> Array:
    """log p(x_1:n) = B(chi', nu') - B(chi, nu) + sum_i log h(x_i)."""
    raise NotImplementedError  # Module 5, Step 4


def posterior_predictive_logpdf(post: ConjugatePrior, x: Array) -> Array:
    """log p(x_new | x_1:n) = B(chi' + t(x), nu' + 1) - B(chi', nu') + log h(x)."""
    raise NotImplementedError  # Module 5, Step 4


# --------------------------------------------------------------------------------------
# Step 5: streaming updates and a decision
# --------------------------------------------------------------------------------------
def sequential_update(prior: ConjugatePrior, xs: Array) -> tuple[ConjugatePrior, Array, Array]:
    """Process xs one at a time with lax.scan.

    Returns (final posterior, chi_trajectory [n, dim], nu_trajectory [n])."""
    raise NotImplementedError  # Module 5, Step 5


def prob_b_beats_a(post_a: BetaBernoulli, post_b: BetaBernoulli, key: Array, n: int) -> Array:
    """P(p_B > p_A | data) by Monte Carlo from the two Beta posteriors."""
    raise NotImplementedError  # Module 5, Step 5
