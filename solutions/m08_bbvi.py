"""Module 8 — A black-box variational inference engine.

Pieces: elementwise bijectors with log-determinant Jacobians (to put constrained
parameters on R^d), a hand-written Adam, a mean-field Gaussian family, a reparameterised
ELBO optimiser `fit`, and a stochastic-VI variant `fit_svi` that subsamples data and
rescales the minibatch log-likelihood by N / B.

Public API (used by Modules 11, 13, 14):
    MeanFieldGaussian(dim): init_params(), sample(key, params, n), log_prob(params, z), entropy(params)
    fit(log_joint, dim, key, n_steps, lr=0.02, n_samples=8, init_params=None) -> (params, elbo_trace)
    fit_svi(log_prior, log_lik_fn, data, dim, key, n_steps, batch_size, lr=0.02, n_samples=8, init_params=None)
    Identity, Exp, Softplus, Sigmoid, Chain  with forward(x), inverse(y), log_det_jacobian(x)
    adam_init(params), adam_update(grads, state, params, lr, b1=0.9, b2=0.999, eps=1e-8)
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Params = dict[str, Array]


# --------------------------------------------------------------------------------------
# Step 1: bijectors
# --------------------------------------------------------------------------------------
class Bijector:
    """Elementwise smooth bijection R -> support. `log_det_jacobian(x)` is
    sum_i log |d forward / dx_i| evaluated at x (the unconstrained point)."""

    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def inverse(self, y: Array) -> Array:
        raise NotImplementedError

    def log_det_jacobian(self, x: Array) -> Array:
        raise NotImplementedError


class Identity(Bijector):
    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def log_det_jacobian(self, x):
        return jnp.zeros(())


class Exp(Bijector):
    """R -> (0, inf)."""

    def forward(self, x):
        # [m08 step 1]
        return jnp.exp(x)

    def inverse(self, y):
        # [m08 step 1]
        return jnp.log(y)

    def log_det_jacobian(self, x):
        # [m08 step 1]
        return jnp.sum(x)


class Softplus(Bijector):
    """R -> (0, inf), y = log(1 + e^x). Inverse: x = y + log(-expm1(-y))."""

    def forward(self, x):
        # [m08 step 1]
        return jax.nn.softplus(x)

    def inverse(self, y):
        # [m08 step 1]
        return y + jnp.log(-jnp.expm1(-y))

    def log_det_jacobian(self, x):
        # [m08 step 1]
        return jnp.sum(jax.nn.log_sigmoid(x))


class Sigmoid(Bijector):
    """R -> (0, 1). d sigmoid/dx = s(x) s(-x)."""

    def forward(self, x):
        # [m08 step 1]
        return jax.nn.sigmoid(x)

    def inverse(self, y):
        # [m08 step 1]
        return jnp.log(y) - jnp.log1p(-y)

    def log_det_jacobian(self, x):
        # [m08 step 1]
        return jnp.sum(jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x))


class Chain(Bijector):
    """Apply bijectors in order: forward = b_n o ... o b_1."""

    def __init__(self, *bijectors: Bijector):
        self.bijectors = bijectors

    def forward(self, x):
        # [m08 step 1]
        for b in self.bijectors:
            x = b.forward(x)
        return x

    def inverse(self, y):
        # [m08 step 1]
        for b in reversed(self.bijectors):
            y = b.inverse(y)
        return y

    def log_det_jacobian(self, x):
        # [m08 step 1]
        total = jnp.zeros(())
        for b in self.bijectors:
            total = total + b.log_det_jacobian(x)
            x = b.forward(x)
        return total


# --------------------------------------------------------------------------------------
# Step 2: Adam on pytrees
# --------------------------------------------------------------------------------------
def adam_init(params):
    """State (m, v, t) with m, v zero pytrees like params and t = 0."""
    # [m08 step 2]
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), dtype=jnp.int32)}


def adam_update(grads, state, params, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    """One Adam ASCENT step (we maximise the ELBO): params + lr * m_hat / (sqrt(v_hat) + eps).
    Returns (new_params, new_state)."""
    # [m08 step 2]
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g**2, state["v"], grads)
    tf = t.astype(jnp.float64)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**tf), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**tf), v)
    new_params = jax.tree_util.tree_map(lambda p, mh, vh: p + lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return new_params, {"m": m, "v": v, "t": t}


# --------------------------------------------------------------------------------------
# Step 3: mean-field Gaussian family
# --------------------------------------------------------------------------------------
class MeanFieldGaussian:
    """q(z) = N(loc, diag(exp(log_scale))^2) on R^dim."""

    def __init__(self, dim: int):
        self.dim = dim

    def init_params(self) -> Params:
        return {"loc": jnp.zeros(self.dim), "log_scale": jnp.zeros(self.dim)}

    def sample(self, key: Array, params: Params, n: int) -> Array:
        """Reparameterised: loc + exp(log_scale) * eps, eps ~ N(0, I). Shape [n, dim]."""
        # [m08 step 3]
        eps = jax.random.normal(key, (n, self.dim))
        return params["loc"] + jnp.exp(params["log_scale"]) * eps

    def log_prob(self, params: Params, z: Array) -> Array:
        # [m08 step 3]
        return jnp.sum(-0.5 * ((z - params["loc"]) * jnp.exp(-params["log_scale"])) ** 2 - params["log_scale"] - 0.5 * jnp.log(2 * jnp.pi))

    def entropy(self, params: Params) -> Array:
        """sum_i [ log_scale_i + 1/2 log(2 pi e) ]."""
        # [m08 step 3]
        return jnp.sum(params["log_scale"]) + 0.5 * self.dim * jnp.log(2 * jnp.pi * jnp.e)


# --------------------------------------------------------------------------------------
# Step 4: the optimiser
# --------------------------------------------------------------------------------------
def elbo_estimate(key: Array, params: Params, family: MeanFieldGaussian, log_joint: Callable[[Array], Array], n_samples: int) -> Array:
    """mean_s log_joint(z_s) + H[q], z_s reparameterised. Entropy is analytic."""
    # [m08 step 4]
    z = family.sample(key, params, n_samples)
    return jnp.mean(jax.vmap(log_joint)(z)) + family.entropy(params)


def fit(log_joint: Callable[[Array], Array], dim: int, key: Array, n_steps: int, lr: float = 0.02, n_samples: int = 8, init_params: Params | None = None) -> tuple[Params, Array]:
    """Maximise the reparameterised ELBO with Adam for n_steps using lax.scan.

    Returns (params, elbo_trace [n_steps]) where elbo_trace[t] is the single-batch ELBO
    estimate used at step t."""
    # [m08 step 4]
    family = MeanFieldGaussian(dim)
    params = family.init_params() if init_params is None else init_params
    state = adam_init(params)
    keys = jax.random.split(key, n_steps)

    def body(carry, k):
        params, state = carry
        value, grads = jax.value_and_grad(elbo_estimate, argnums=1)(k, params, family, log_joint, n_samples)
        params, state = adam_update(grads, state, params, lr)
        return (params, state), value

    (params, _), trace = lax.scan(body, (params, state), keys)
    return params, trace


# --------------------------------------------------------------------------------------
# Step 5: stochastic VI with data subsampling
# --------------------------------------------------------------------------------------
def fit_svi(log_prior: Callable[[Array], Array], log_lik_fn: Callable[[Array, Array, Array], Array], data: tuple[Array, Array], dim: int, key: Array, n_steps: int, batch_size: int, lr: float = 0.02, n_samples: int = 8, init_params: Params | None = None) -> tuple[Params, Array]:
    """Stochastic VI: each step draws a minibatch of B row indices uniformly (with
    replacement, as in Hoffman et al. 2013), forms
        log_joint_B(z) = log_prior(z) + (N / B) * log_lik_fn(z, X_B, y_B),
    and takes one Adam step on the reparameterised ELBO. log_lik_fn(z, X, y) returns the
    summed log-likelihood of the rows given. Returns (params, elbo_trace)."""
    # [m08 step 5]
    X, y = data
    N = X.shape[0]
    family = MeanFieldGaussian(dim)
    params = family.init_params() if init_params is None else init_params
    state = adam_init(params)
    keys = jax.random.split(key, n_steps)

    def body(carry, k):
        params, state = carry
        k_idx, k_z = jax.random.split(k)
        idx = jax.random.randint(k_idx, (batch_size,), 0, N)
        Xb, yb = X[idx], y[idx]
        lj = lambda z: log_prior(z) + (N / batch_size) * log_lik_fn(z, Xb, yb)
        value, grads = jax.value_and_grad(elbo_estimate, argnums=1)(k_z, params, family, lj, n_samples)
        params, state = adam_update(grads, state, params, lr)
        return (params, state), value

    (params, _), trace = lax.scan(body, (params, state), keys)
    return params, trace
