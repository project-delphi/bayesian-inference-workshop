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
    raise NotImplementedError  # Module 2, Step 1


def pushforward_logpdf(log_p_x: Callable[[Array], Array], f_inv: Callable[[Array], Array], y: Array) -> Array:
    """log density of y = f(x) given log p_x and the inverse map f_inv, for vector y.

    log p_y(y) = log p_x(f_inv(y)) + log |det J_{f_inv}(y)|. Use jax.jacfwd for the
    Jacobian and jnp.linalg.slogdet for the determinant.
    """
    raise NotImplementedError  # Module 2, Step 2


def grad_logdet(A: Array) -> Array:
    """Closed-form gradient of log det A with respect to A (A square, invertible).
    Do not use autodiff here; the test compares you against it."""
    raise NotImplementedError  # Module 2, Step 3


def grad_quadratic_form_x(A: Array, x: Array) -> Array:
    """Closed-form gradient of x^T A x with respect to x, for general (non-symmetric) A."""
    raise NotImplementedError  # Module 2, Step 3


def grad_trace_inv(A: Array, B: Array) -> Array:
    """Closed-form gradient of tr(A^{-1} B) with respect to A."""
    raise NotImplementedError  # Module 2, Step 3


def kl_gaussians(mu0: Array, cov0: Array, mu1: Array, cov1: Array) -> Array:
    """KL( N(mu0, cov0) || N(mu1, cov1) ) in closed form for d-dimensional Gaussians.
    Use Cholesky factors and triangular solves; no explicit inverses."""
    raise NotImplementedError  # Module 2, Step 4


def quiz_answers() -> dict[str, str]:
    """Answers to the eight conceptual questions on the Module 2 page.

    Return a dict mapping "q1".."q8" to a single lower-case letter. The test compares
    hashes, so the answers are not readable from the test file.
    """
    raise NotImplementedError  # Module 2, Step 5
