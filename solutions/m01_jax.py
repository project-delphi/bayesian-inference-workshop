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
    # [m01 step 1]
    m = jnp.max(a, axis=axis, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    s = jnp.sum(jnp.exp(a - m), axis=axis, keepdims=True)
    out = jnp.log(s) + m
    return out if axis is None and out.ndim == 0 else jnp.squeeze(out, axis=axis) if axis is not None else jnp.squeeze(out)


def gaussian_logpdf(x: Array, mu: Array, cov: Array) -> Array:
    """Log density of a single d-vector x under N(mu, cov).

    Use a Cholesky factor; never form cov^{-1} explicitly.
    """
    # [m01 step 2]
    d = x.shape[-1]
    L = jnp.linalg.cholesky(cov)
    r = jax.scipy.linalg.solve_triangular(L, x - mu, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + r @ r)


def batched_gaussian_logpdf(xs: Array, mu: Array, cov: Array) -> Array:
    """Log density of each row of xs [n, d] under N(mu, cov), via jax.vmap over rows.

    No Python loops, no explicit broadcasting of cov.
    """
    # [m01 step 2]
    return jax.vmap(gaussian_logpdf, in_axes=(0, None, None))(xs, mu, cov)


def newton_step(f: Callable[[Array], Array], x: Array) -> Array:
    """One Newton step x - H^{-1} g for a scalar function f of a vector x, using
    jax.grad and jax.hessian. Solve the linear system; do not invert H."""
    # [m01 step 3]
    g = jax.grad(f)(x)
    H = jax.hessian(f)(x)
    return x - jnp.linalg.solve(H, g)


def newton_minimize(f: Callable[[Array], Array], x0: Array, n_steps: int) -> tuple[Array, Array]:
    """Run `n_steps` Newton steps with lax.scan.

    Returns (x_final, trajectory) where trajectory has shape [n_steps, d] and
    trajectory[i] is the iterate after step i+1. The whole thing must be jit-able with
    `n_steps` static.
    """
    # [m01 step 3]

    def body(x, _):
        x_new = newton_step(f, x)
        return x_new, x_new

    x_final, traj = lax.scan(body, x0, None, length=n_steps)
    return x_final, traj


def sample_gaussian_mixture(key: Array, weights: Array, means: Array, scales: Array, n: int) -> tuple[Array, Array]:
    """Draw n samples from a 1-D Gaussian mixture.

    weights [K] sum to one; means [K]; scales [K] > 0. Split `key` once into a key for
    the component labels and a key for the Gaussian noise. Returns (x [n], z [n]) where
    z holds the integer component labels. Use jax.random.categorical for z.
    """
    # [m01 step 4]
    k_z, k_x = jax.random.split(key)
    z = jax.random.categorical(k_z, jnp.log(weights), shape=(n,))
    eps = jax.random.normal(k_x, (n,))
    x = means[z] + scales[z] * eps
    return x, z


def ar1_simulate(key: Array, phi: float, sigma: float, n: int, x0: float = 0.0) -> Array:
    """Simulate x_t = phi * x_{t-1} + sigma * eps_t for t = 1..n with lax.scan.

    Draw all n noise variables up front with a single call to jax.random.normal, then
    scan over them. Returns x [n] (excluding x0).
    """
    # [m01 step 5]
    eps = jax.random.normal(key, (n,))

    def body(x_prev, e):
        x = phi * x_prev + sigma * e
        return x, x

    _, xs = lax.scan(body, jnp.asarray(x0, dtype=eps.dtype), eps)
    return xs


def cumulative_logsumexp(a: Array) -> Array:
    """out[t] = logsumexp(a[:t+1]) for a 1-D array, computed in one lax.scan pass using
    the running maximum (no O(n^2) work, no overflow). Must return -inf, not NaN, while
    every entry so far is -inf, and the outputs after that must be unaffected."""
    # [m01 step 5]

    def body(carry, a_t):
        m, s = carry  # running max and sum of exp(a - m)
        m_new = jnp.maximum(m, a_t)
        # While every input so far is -inf, m_new is -inf and m - m_new would be NaN.
        # Rescale by a finite reference instead; carry the true maximum so later
        # very negative inputs are not underflowed against a made-up reference.
        ref = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
        s_new = s * jnp.exp(m - ref) + jnp.exp(a_t - ref)
        return (m_new, s_new), jnp.log(s_new) + ref

    init = (jnp.asarray(-jnp.inf, dtype=a.dtype), jnp.asarray(0.0, dtype=a.dtype))
    _, out = lax.scan(body, init, a)
    return out


def tree_sgd_step(params, grads, lr: float):
    """Return params - lr * grads for arbitrary nested pytrees of arrays (dicts, lists,
    tuples). Use jax.tree_util.tree_map; the structure must be preserved exactly."""
    # [m01 step 6]
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def tree_l2_norm(tree) -> Array:
    """sqrt of the sum of squares of every leaf in a pytree."""
    # [m01 step 6]
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(l**2) for l in leaves))
