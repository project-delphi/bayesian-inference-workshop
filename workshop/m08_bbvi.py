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
        raise NotImplementedError  # Module 8, Step 1

    def inverse(self, y):
        raise NotImplementedError  # Module 8, Step 1

    def log_det_jacobian(self, x):
        raise NotImplementedError  # Module 8, Step 1


class Softplus(Bijector):
    """R -> (0, inf), y = log(1 + e^x). Inverse: x = y + log(-expm1(-y))."""

    def forward(self, x):
        raise NotImplementedError  # Module 8, Step 1

    def inverse(self, y):
        raise NotImplementedError  # Module 8, Step 1

    def log_det_jacobian(self, x):
        raise NotImplementedError  # Module 8, Step 1


class Sigmoid(Bijector):
    """R -> (0, 1). d sigmoid/dx = s(x) s(-x)."""

    def forward(self, x):
        raise NotImplementedError  # Module 8, Step 1

    def inverse(self, y):
        raise NotImplementedError  # Module 8, Step 1

    def log_det_jacobian(self, x):
        raise NotImplementedError  # Module 8, Step 1


class Chain(Bijector):
    """Apply bijectors in order: forward = b_n o ... o b_1."""

    def __init__(self, *bijectors: Bijector):
        self.bijectors = bijectors

    def forward(self, x):
        raise NotImplementedError  # Module 8, Step 1

    def inverse(self, y):
        raise NotImplementedError  # Module 8, Step 1

    def log_det_jacobian(self, x):
        raise NotImplementedError  # Module 8, Step 1


# --------------------------------------------------------------------------------------
# Step 2: Adam on pytrees
# --------------------------------------------------------------------------------------
def adam_init(params):
    """State (m, v, t) with m, v zero pytrees like params and t = 0."""
    raise NotImplementedError  # Module 8, Step 2


def adam_update(grads, state, params, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    """One Adam ASCENT step (we maximise the ELBO): params + lr * m_hat / (sqrt(v_hat) + eps).
    Returns (new_params, new_state)."""
    raise NotImplementedError  # Module 8, Step 2


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
        raise NotImplementedError  # Module 8, Step 3

    def log_prob(self, params: Params, z: Array) -> Array:
        raise NotImplementedError  # Module 8, Step 3

    def entropy(self, params: Params) -> Array:
        """sum_i [ log_scale_i + 1/2 log(2 pi e) ]."""
        raise NotImplementedError  # Module 8, Step 3


# --------------------------------------------------------------------------------------
# Step 4: the optimiser
# --------------------------------------------------------------------------------------
def elbo_estimate(key: Array, params: Params, family: MeanFieldGaussian, log_joint: Callable[[Array], Array], n_samples: int) -> Array:
    """mean_s log_joint(z_s) + H[q], z_s reparameterised. Entropy is analytic."""
    raise NotImplementedError  # Module 8, Step 4


def fit(log_joint: Callable[[Array], Array], dim: int, key: Array, n_steps: int, lr: float = 0.02, n_samples: int = 8, init_params: Params | None = None) -> tuple[Params, Array]:
    """Maximise the reparameterised ELBO with Adam for n_steps using lax.scan.

    Returns (params, elbo_trace [n_steps]) where elbo_trace[t] is the single-batch ELBO
    estimate used at step t."""
    raise NotImplementedError  # Module 8, Step 4


# --------------------------------------------------------------------------------------
# Step 5: stochastic VI with data subsampling
# --------------------------------------------------------------------------------------
def fit_svi(log_prior: Callable[[Array], Array], log_lik_fn: Callable[[Array, Array, Array], Array], data: tuple[Array, Array], dim: int, key: Array, n_steps: int, batch_size: int, lr: float = 0.02, n_samples: int = 8, init_params: Params | None = None) -> tuple[Params, Array]:
    """Stochastic VI: each step draws a minibatch of B row indices uniformly (with
    replacement, as in Hoffman et al. 2013), forms
        log_joint_B(z) = log_prior(z) + (N / B) * log_lik_fn(z, X_B, y_B),
    and takes one Adam step on the reparameterised ELBO. log_lik_fn(z, X, y) returns the
    summed log-likelihood of the rows given. Returns (params, elbo_trace)."""
    raise NotImplementedError  # Module 8, Step 5
