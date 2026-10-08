"""Module 6 — The ELBO and coordinate-ascent variational inference (CAVI).

Model (Bayesian Gaussian mixture with known isotropic variance sigma^2):

    pi ~ Dirichlet(alpha0 1_K)
    mu_k ~ N(0, sigma0^2 I_d)                     k = 1..K
    z_i ~ Categorical(pi),  x_i | z_i ~ N(mu_{z_i}, sigma^2 I_d)   i = 1..n

Mean-field family q(pi) q(mu) q(z) = Dir(pi | alpha) prod_k N(mu_k | m_k, s_k^2 I)
prod_i Cat(z_i | r_i). Each CAVI update sets one factor to
exp(E_{-j}[log p(x, z, pi, mu)]) normalised, which for conjugate factors is a
natural-parameter addition of expected sufficient statistics (Module 5).

Conventions: x [n, d]; r [n, K]; alpha [K]; m [K, d]; s2 [K] (one scalar variance per
component, isotropic).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma, gammaln

from .m04_kl_bregman import kl_dirichlet

Array = jax.Array


class GMMPrior(NamedTuple):
    alpha0: float  # Dirichlet concentration (symmetric)
    sigma0: float  # prior SD of component means
    sigma: float  # known likelihood SD


class VarParams(NamedTuple):
    r: Array  # [n, K] responsibilities
    alpha: Array  # [K] Dirichlet
    m: Array  # [K, d] Gaussian means
    s2: Array  # [K] Gaussian variances (isotropic)


# --------------------------------------------------------------------------------------
# Step 1: ELBO in closed form
# --------------------------------------------------------------------------------------
def expected_log_lik(x: Array, r: Array, m: Array, s2: Array, sigma: float) -> Array:
    """sum_i sum_k r_ik E_q[log N(x_i | mu_k, sigma^2 I)].

    E_q[(x_i - mu_k)^2] = |x_i - m_k|^2 + d s2_k."""
    raise NotImplementedError  # Module 6, Step 1


def expected_log_prior_z(r: Array, alpha: Array) -> Array:
    """sum_i sum_k r_ik E_q[log pi_k] with E[log pi_k] = digamma(alpha_k) - digamma(sum alpha)."""
    raise NotImplementedError  # Module 6, Step 1


def categorical_entropy(r: Array) -> Array:
    """-sum_i sum_k r_ik log r_ik, treating 0 log 0 as 0."""
    raise NotImplementedError  # Module 6, Step 1


def kl_gaussian_means(m: Array, s2: Array, sigma0: float) -> Array:
    """sum_k KL( N(m_k, s2_k I) || N(0, sigma0^2 I) ), isotropic, d-dimensional."""
    raise NotImplementedError  # Module 6, Step 1


def elbo(x: Array, q: VarParams, prior: GMMPrior) -> Array:
    """ELBO = E[log lik] + E[log p(z|pi)] + H[q(z)] - KL(q(pi)||p(pi)) - KL(q(mu)||p(mu))."""
    raise NotImplementedError  # Module 6, Step 1


# --------------------------------------------------------------------------------------
# Step 2: responsibilities
# --------------------------------------------------------------------------------------
def update_responsibilities(x: Array, alpha: Array, m: Array, s2: Array, sigma: float) -> Array:
    """log r_ik = E[log pi_k] - E[|x_i - mu_k|^2] / (2 sigma^2) + const, normalised per row
    with a stable log-softmax."""
    raise NotImplementedError  # Module 6, Step 2


# --------------------------------------------------------------------------------------
# Step 3: conjugate factor updates
# --------------------------------------------------------------------------------------
def update_dirichlet(r: Array, alpha0: float) -> Array:
    """alpha_k = alpha0 + sum_i r_ik  (natural parameter of Dir plus expected counts)."""
    raise NotImplementedError  # Module 6, Step 3


def update_gaussian_means(x: Array, r: Array, sigma0: float, sigma: float) -> tuple[Array, Array]:
    """Precision-weighted update: 1/s2_k = 1/sigma0^2 + N_k/sigma^2,
    m_k = s2_k * (sum_i r_ik x_i) / sigma^2 with N_k = sum_i r_ik."""
    raise NotImplementedError  # Module 6, Step 3


# --------------------------------------------------------------------------------------
# Step 4: the CAVI loop
# --------------------------------------------------------------------------------------
def init_responsibilities(key: Array, x: Array, K: int) -> Array:
    """Deterministic k-means++-style seeding followed by one hard assignment, then
    softened (0.9 on the assigned cluster)."""
    n = x.shape[0]
    first = jax.random.randint(key, (), 0, n)
    centres = jnp.zeros((K, x.shape[1])).at[0].set(x[first])

    def body(c, k):
        d2 = jnp.min(jnp.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=-1) + jnp.where(jnp.arange(K) >= k, 1e12, 0.0)[None, :], axis=1)
        idx = jnp.argmax(d2)  # farthest point (deterministic variant)
        return c.at[k].set(x[idx]), None

    centres, _ = lax.scan(body, centres, jnp.arange(1, K))
    assign = jnp.argmin(jnp.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=-1), axis=1)
    r = jax.nn.one_hot(assign, K) * 0.9 + 0.1 / K
    return r


def cavi_step(x: Array, q: VarParams, prior: GMMPrior) -> VarParams:
    """One sweep: update q(pi), q(mu) from r, then r from the new factors."""
    raise NotImplementedError  # Module 6, Step 4


def cavi(x: Array, r0: Array, prior: GMMPrior, n_iter: int) -> tuple[VarParams, Array]:
    """Run n_iter CAVI sweeps from initial responsibilities r0 with lax.scan.

    Returns (final VarParams, elbo_trace [n_iter]) where elbo_trace[t] is the ELBO after
    sweep t+1."""
    raise NotImplementedError  # Module 6, Step 4


# --------------------------------------------------------------------------------------
# Step 5: ELBO as a model-selection score
# --------------------------------------------------------------------------------------
def elbo_by_k(key: Array, x: Array, ks: tuple[int, ...], prior: GMMPrior, n_iter: int = 100) -> Array:
    """Fit CAVI for each K in ks (same key for seeding) and return the final ELBOs."""
    raise NotImplementedError  # Module 6, Step 5
