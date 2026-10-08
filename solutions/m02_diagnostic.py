"""Module 2 — Diagnostic.

Auto-graded check of the mathematics the rest of the workshop assumes: Gaussian
conditioning, change of variables, matrix calculus, KL between Gaussians, and a short
conceptual quiz. Nothing here is taught on the site; if a step fails, the module page
lists what to review.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def gaussian_condition(mu: Array, cov: Array, obs_idx: Array, x_obs: Array) -> tuple[Array, Array]:
    """Condition x ~ N(mu, cov) on x[obs_idx] = x_obs.

    Returns (mu_cond, cov_cond) for the remaining coordinates, in their original order.
    obs_idx is an integer array of observed coordinate positions.
    """
    # [m02 step 1]
    d = mu.shape[0]
    mask = jnp.zeros(d, dtype=bool).at[obs_idx].set(True)
    free_idx = jnp.nonzero(~mask, size=d - obs_idx.shape[0])[0]
    S_ff = cov[jnp.ix_(free_idx, free_idx)]
    S_fo = cov[jnp.ix_(free_idx, obs_idx)]
    S_oo = cov[jnp.ix_(obs_idx, obs_idx)]
    K = jnp.linalg.solve(S_oo, S_fo.T).T  # S_fo S_oo^{-1}
    mu_c = mu[free_idx] + K @ (x_obs - mu[obs_idx])
    cov_c = S_ff - K @ S_fo.T
    return mu_c, cov_c


def pushforward_logpdf(log_p_x: Callable[[Array], Array], f_inv: Callable[[Array], Array], y: Array) -> Array:
    """log density of y = f(x) given log p_x and the inverse map f_inv, for vector y.

    log p_y(y) = log p_x(f_inv(y)) + log |det J_{f_inv}(y)|. Use jax.jacfwd for the
    Jacobian and jnp.linalg.slogdet for the determinant.
    """
    # [m02 step 2]
    x = f_inv(y)
    J = jax.jacfwd(f_inv)(y)
    _, logabsdet = jnp.linalg.slogdet(J)
    return log_p_x(x) + logabsdet


def grad_logdet(A: Array) -> Array:
    """Closed-form gradient of log det A with respect to A (A square, invertible).
    Do not use autodiff here; the test compares you against it."""
    # [m02 step 3]
    return jnp.linalg.inv(A).T


def grad_quadratic_form_x(A: Array, x: Array) -> Array:
    """Closed-form gradient of x^T A x with respect to x, for general (non-symmetric) A."""
    # [m02 step 3]
    return (A + A.T) @ x


def grad_trace_inv(A: Array, B: Array) -> Array:
    """Closed-form gradient of tr(A^{-1} B) with respect to A."""
    # [m02 step 3]
    Ainv = jnp.linalg.inv(A)
    return -(Ainv @ B @ Ainv).T


def kl_gaussians(mu0: Array, cov0: Array, mu1: Array, cov1: Array) -> Array:
    """KL( N(mu0, cov0) || N(mu1, cov1) ) in closed form for d-dimensional Gaussians.
    Use Cholesky factors and triangular solves; no explicit inverses."""
    # [m02 step 4]
    d = mu0.shape[0]
    L0 = jnp.linalg.cholesky(cov0)
    L1 = jnp.linalg.cholesky(cov1)
    logdet0 = 2 * jnp.sum(jnp.log(jnp.diag(L0)))
    logdet1 = 2 * jnp.sum(jnp.log(jnp.diag(L1)))
    M = jax.scipy.linalg.solve_triangular(L1, L0, lower=True)  # L1^{-1} L0
    tr_term = jnp.sum(M**2)  # tr(cov1^{-1} cov0)
    r = jax.scipy.linalg.solve_triangular(L1, mu1 - mu0, lower=True)
    quad = r @ r
    return 0.5 * (tr_term + quad - d + logdet1 - logdet0)


def quiz_answers() -> dict[str, str]:
    """Answers to the eight conceptual questions on the Module 2 page.

    Return a dict mapping "q1".."q8" to a single lower-case letter. The test compares
    hashes, so the answers are not readable from the test file.
    """
    # [m02 step 5]
    return {
        "q1": "b",  # Jensen for convex f
        "q2": "c",  # log-partition convexity
        "q3": "a",  # Hessian of A
        "q4": "b",  # change of variables Jacobian
        "q5": "c",  # expected score
        "q6": "a",  # covariance of a linear map
        "q7": "a",  # mutual information as a KL
        "q8": "b",  # gradient of logsumexp
    }
