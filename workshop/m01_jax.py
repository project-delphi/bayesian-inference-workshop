"""Module 1 — JAX warm-up.

Functional numerics in JAX: stable reductions, autodiff, vectorisation, explicit PRNG
keys, `lax.scan` and pytrees. Everything later in the workshop uses these six idioms.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


def logsumexp(a: Array, axis: int | None = None) -> Array:
    """Numerically stable log(sum(exp(a))) along `axis`.

    Must not overflow for entries like 1e4 and must return -inf (not NaN) when every
    entry along the axis is -inf. Do not call jax.nn.logsumexp or jax.scipy.
    """
    raise NotImplementedError  # Module 1, Step 1


def gaussian_logpdf(x: Array, mu: Array, cov: Array) -> Array:
    """Log density of a single d-vector x under N(mu, cov).

    Use a Cholesky factor; never form cov^{-1} explicitly.
    """
    raise NotImplementedError  # Module 1, Step 2


def batched_gaussian_logpdf(xs: Array, mu: Array, cov: Array) -> Array:
    """Log density of each row of xs [n, d] under N(mu, cov), via jax.vmap over rows.

    No Python loops, no explicit broadcasting of cov.
    """
    raise NotImplementedError  # Module 1, Step 2


def newton_step(f: Callable[[Array], Array], x: Array) -> Array:
    """One Newton step x - H^{-1} g for a scalar function f of a vector x, using
    jax.grad and jax.hessian. Solve the linear system; do not invert H."""
    raise NotImplementedError  # Module 1, Step 3


def newton_minimize(f: Callable[[Array], Array], x0: Array, n_steps: int) -> tuple[Array, Array]:
    """Run `n_steps` Newton steps with lax.scan.

    Returns (x_final, trajectory) where trajectory has shape [n_steps, d] and
    trajectory[i] is the iterate after step i+1. The whole thing must be jit-able with
    `n_steps` static.
    """
    raise NotImplementedError  # Module 1, Step 3


def sample_gaussian_mixture(key: Array, weights: Array, means: Array, scales: Array, n: int) -> tuple[Array, Array]:
    """Draw n samples from a 1-D Gaussian mixture.

    weights [K] sum to one; means [K]; scales [K] > 0. Split `key` once into a key for
    the component labels and a key for the Gaussian noise. Returns (x [n], z [n]) where
    z holds the integer component labels. Use jax.random.categorical for z.
    """
    raise NotImplementedError  # Module 1, Step 4


def ar1_simulate(key: Array, phi: float, sigma: float, n: int, x0: float = 0.0) -> Array:
    """Simulate x_t = phi * x_{t-1} + sigma * eps_t for t = 1..n with lax.scan.

    Draw all n noise variables up front with a single call to jax.random.normal, then
    scan over them. Returns x [n] (excluding x0).
    """
    raise NotImplementedError  # Module 1, Step 5


def cumulative_logsumexp(a: Array) -> Array:
    """out[t] = logsumexp(a[:t+1]) for a 1-D array, computed in one lax.scan pass using
    the running maximum (no O(n^2) work, no overflow). Must return -inf, not NaN, while
    every entry so far is -inf, and the outputs after that must be unaffected."""
    raise NotImplementedError  # Module 1, Step 5


def tree_sgd_step(params, grads, lr: float):
    """Return params - lr * grads for arbitrary nested pytrees of arrays (dicts, lists,
    tuples). Use jax.tree_util.tree_map; the structure must be preserved exactly."""
    raise NotImplementedError  # Module 1, Step 6


def tree_l2_norm(tree) -> Array:
    """sqrt of the sum of squares of every leaf in a pytree."""
    raise NotImplementedError  # Module 1, Step 6
