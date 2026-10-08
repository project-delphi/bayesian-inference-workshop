"""Module 2b — Bayesian inference from first principles.

One model, every route to its posterior. The model is twelve qPCR log2 fold-change
replicates with unknown mean mu and unknown precision lambda:

    mu     ~ Normal(0, 10^2)
    lambda ~ Gamma(shape=2, rate=0.5)
    x_i    ~ Normal(mu, 1 / lambda)      i = 1..n

All inference code works in the unconstrained parameter theta = (mu, log lambda). The
change of variables lambda = exp(log lambda) contributes a Jacobian term +log lambda to
the log prior density over theta; Step 1 makes it explicit.

The module computes the posterior four ways: on a grid (exact up to discretisation),
with a Gaussian fitted by minimising KL (variational inference in miniature), with a
Laplace approximation (a Gaussian at the mode), and with a random-walk Metropolis
sampler (MCMC in miniature). Days 1 to 4 build industrial versions of each.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

Array = jax.Array

PRIOR_MU_SD = 10.0
PRIOR_LAM_SHAPE = 2.0
PRIOR_LAM_RATE = 0.5


# --------------------------------------------------------------------------------------
# Step 1: the model as three log densities
# --------------------------------------------------------------------------------------
def log_prior_constrained(mu: Array, lam: Array) -> Array:
    """log p(mu) + log p(lambda) with mu ~ N(0, 10^2), lambda ~ Gamma(2, rate 0.5),
    as a density over (mu, lambda). Provided."""
    log_p_mu = -0.5 * (mu / PRIOR_MU_SD) ** 2 - jnp.log(PRIOR_MU_SD) - 0.5 * jnp.log(2 * jnp.pi)
    a, b = PRIOR_LAM_SHAPE, PRIOR_LAM_RATE
    log_p_lam = a * jnp.log(b) - gammaln(a) + (a - 1) * jnp.log(lam) - b * lam
    return log_p_mu + log_p_lam


def log_prior(theta: Array) -> Array:
    """log prior density over theta = (mu, log lambda).

    Equals log_prior_constrained(mu, lambda) + log lambda: the second term is the log
    absolute derivative d lambda / d(log lambda) = lambda, the Jacobian of the change of
    variables, so that the density integrates to one over theta.
    """
    raise NotImplementedError  # Module 2b, Step 1


def log_likelihood(theta: Array, x: Array) -> Array:
    """sum_i log Normal(x_i | mu, 1 / lambda) for theta = (mu, log lambda)."""
    raise NotImplementedError  # Module 2b, Step 1


def log_joint(theta: Array, x: Array) -> Array:
    """log p(x, theta) = log_prior(theta) + log_likelihood(theta, x). The unnormalised log
    posterior: it differs from log p(theta | x) by the constant log p(x)."""
    raise NotImplementedError  # Module 2b, Step 1


# --------------------------------------------------------------------------------------
# Step 2: the posterior on a grid
# --------------------------------------------------------------------------------------
def grid_log_posterior(log_joint_fn: Callable[[Array, Array], Array], x: Array, mu_grid: Array, loglam_grid: Array) -> tuple[Array, Array]:
    """Evaluate log p(x, theta) on the product grid and normalise.

    Returns (log_post [M, L], log_evidence) where log_post[i, j] is the normalised log
    posterior density at (mu_grid[i], loglam_grid[j]) and log_evidence approximates
    log p(x) = log integral p(x, theta) d theta by the Riemann sum with cell area
    d_mu * d_loglam. Use jax.vmap twice and logsumexp; never exponentiate before
    subtracting the maximum.
    """
    raise NotImplementedError  # Module 2b, Step 2


def grid_moments(log_post: Array, mu_grid: Array, loglam_grid: Array) -> tuple[Array, Array]:
    """Posterior mean [2] and covariance [2, 2] of theta from the normalised grid.

    Cell weights are exp(log_post) * cell area; they sum to one."""
    raise NotImplementedError  # Module 2b, Step 2


# --------------------------------------------------------------------------------------
# Step 3: prediction is an expectation under the posterior
# --------------------------------------------------------------------------------------
def posterior_predictive_grid(log_post: Array, mu_grid: Array, loglam_grid: Array, x_new: Array) -> Array:
    """log p(x_new | x) = log sum_cells w_cell Normal(x_new | mu_cell, 1 / lambda_cell),
    for a scalar x_new. Compute in the log domain with logsumexp."""
    raise NotImplementedError  # Module 2b, Step 3


# --------------------------------------------------------------------------------------
# Step 4: variational inference in miniature, and the Laplace approximation
# --------------------------------------------------------------------------------------
def gaussian_log_density_diag(theta_pts: Array, q_mean: Array, q_log_sd: Array) -> Array:
    """log N(theta | q_mean, diag(exp(q_log_sd)^2)) for a batch of points [N, 2]. Provided."""
    z = (theta_pts - q_mean) / jnp.exp(q_log_sd)
    return jnp.sum(-0.5 * z**2 - q_log_sd - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def gaussian_kl_to_grid(q_mean: Array, q_log_sd: Array, log_post: Array, mu_grid: Array, loglam_grid: Array) -> Array:
    """KL(q || p) = E_q[log q(theta) - log p(theta | x)] estimated on the grid:
    sum over cells of q(theta_cell) * area * (log q(theta_cell) - log_post_cell).
    q is a diagonal Gaussian with parameters (q_mean [2], q_log_sd [2])."""
    raise NotImplementedError  # Module 2b, Step 4


def gaussian_entropy_diag(q_log_sd: Array) -> Array:
    """Entropy of N(mean, diag(exp(q_log_sd)^2)): sum_k (q_log_sd_k + 1/2 log(2 pi e)). Provided."""
    return jnp.sum(q_log_sd + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def elbo_gaussian(q_mean: Array, q_log_sd: Array, eps: Array, log_joint_fn: Callable[[Array, Array], Array], x: Array) -> Array:
    """Evidence lower bound for a diagonal Gaussian q, estimated with fixed standard-normal
    draws eps [S, 2] through the reparameterisation theta = q_mean + exp(q_log_sd) * eps:

        ELBO = mean_s log p(x, theta_s) + H(q).

    Because eps is fixed, the estimate is a deterministic, differentiable function of the
    variational parameters. ELBO <= log p(x), with equality iff q is the posterior."""
    raise NotImplementedError  # Module 2b, Step 4


def adam_step(params: tuple, grads: tuple, state: tuple, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> tuple[tuple, tuple]:
    """One Adam ascent step on a tuple of arrays. Provided here; Module 8 builds it.

    Adam rescales each coordinate's gradient by a running estimate of its magnitude, so
    the step size is about lr in every direction regardless of curvature. Plain gradient
    ascent with one learning rate diverges here because the posterior is about a
    hundred times more curved in mu than in log lambda."""
    m, v, t = state
    t = t + 1
    m = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads))
    v = tuple(b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, grads))
    mhat = tuple(mi / (1 - b1**t) for mi in m)
    vhat = tuple(vi / (1 - b2**t) for vi in v)
    new_params = tuple(p + lr * mh / (jnp.sqrt(vh) + eps) for p, mh, vh in zip(params, mhat, vhat))
    return new_params, (m, v, t)


def fit_gaussian_vi(log_joint_fn: Callable[[Array, Array], Array], x: Array, eps: Array, init_mean: Array, init_log_sd: Array, n_steps: int = 500, lr: float = 0.05) -> tuple[Array, Array, Array]:
    """Maximise elbo_gaussian over (q_mean, q_log_sd) with jax.grad and adam_step in a
    lax.scan loop. Returns (q_mean, q_log_sd, elbo_trace [n_steps])."""
    raise NotImplementedError  # Module 2b, Step 4


def laplace_approx(log_joint_fn: Callable[[Array, Array], Array], x: Array, theta0: Array, n_newton: int = 20) -> tuple[Array, Array]:
    """Gaussian approximation at the mode.

    Find the maximum of f(theta) = log p(x, theta) by damped Newton's method from theta0:
    direction d = -H^{-1} g, then the largest step in {1, 1/2, ..., 2^-8} that increases
    f (a full Newton step can overshoot where f is not concave). Return (mode, cov) with
    cov = (-H(mode))^{-1}."""
    raise NotImplementedError  # Module 2b, Step 4


# --------------------------------------------------------------------------------------
# Step 5: Markov chain Monte Carlo in miniature, and why grids do not scale
# --------------------------------------------------------------------------------------
def random_walk_metropolis(log_target: Callable[[Array], Array], key: Array, theta0: Array, n_steps: int, step_size: float) -> tuple[Array, Array]:
    """Random-walk Metropolis: propose theta' = theta + step_size * eps, eps ~ N(0, I);
    accept with probability min(1, p(theta') / p(theta)). Returns (samples [n, d],
    acceptance rate). Draw proposal noise and uniforms up front; use lax.scan."""
    raise NotImplementedError  # Module 2b, Step 5


def grid_cost(d: int, k: int) -> int:
    """Number of density evaluations for a grid with k points per dimension in d
    dimensions: k ** d. The reason none of Days 2 to 5 uses a grid."""
    raise NotImplementedError  # Module 2b, Step 5
