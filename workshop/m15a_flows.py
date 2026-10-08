"""Module 15a — Normalising flows as variational families (Capstone, Track A).

A RealNVP flow is a composition of affine coupling layers. Each layer leaves half the
coordinates unchanged and applies an elementwise affine map to the other half whose
shift and log-scale are arbitrary functions (small MLPs) of the unchanged half. The
Jacobian is triangular, so the log-determinant is a sum of the log-scales, and the
inverse is exact. With a standard-normal base density this gives a tractable,
expressive variational family q_phi(x) that is fitted by maximising the same
reparameterised ELBO as Module 8, and that can represent the funnel geometry of the
centred hierarchical model on which mean-field Gaussians fail.

Public API
----------
init_conditioner(key, sizes)  (Module 14's MLP with a zero last layer)
make_masks(dim, n_layers) -> [n_layers, dim]
coupling_forward(params, mask, z) -> (x, log_det); coupling_inverse(params, mask, x) -> (z, log_det)
init_flow(key, dim, n_layers, hidden=32) -> params
flow_forward(params, z) -> (x, log_det); flow_inverse(params, x) -> (z, log_det)
flow_log_prob(params, x); flow_sample(key, params, n) -> (x, log_q)
flow_elbo(key, params, log_prob, n_samples); fit_flow_vi(log_prob, params, key, n_steps, lr=1e-3, n_samples=16)
hmc_reference(key, n_warmup=1000, n_samples=1000) -> centred samples [4 * n_samples, 10]
tail_mass(samples, coord=1, threshold=-1.0)
importance_weighted_elbo(log_prob, params, key, n_samples, k); importance_ess(log_w)
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from .m08_bbvi import adam_init, adam_update
from .m09_hmc import UPLIFT_DIM, uplift_noncentered_log_prob, uplift_noncentered_to_centered
from .m11_diagnostics import run_chains
from .m14_vae import init_mlp, mlp

Array = jax.Array
LogProb = Callable[[Array], Array]

SCALE_BOUND = 2.0  # |log-scale| of every coupling layer is bounded by this (tanh squashing)


# --------------------------------------------------------------------------------------
# Step 1: conditioner MLP and one affine coupling layer
# --------------------------------------------------------------------------------------
def init_conditioner(key: Array, sizes: list[int]) -> list[dict[str, Array]]:
    """Module 14's init_mlp(key, sizes) with the LAST layer's weights set to zero (its
    biases are already zero), so that the conditioner outputs zero shift and log-scale
    and the coupling is the identity at initialisation. Apply it with Module 14's mlp."""
    raise NotImplementedError  # Module 15a, Step 1


def make_masks(dim: int, n_layers: int) -> Array:
    """Alternating binary masks [n_layers, dim]: layer k conditions on the coordinates
    with (index % 2 == k % 2) and transforms the others."""
    raise NotImplementedError  # Module 15a, Step 1


def _shift_log_scale(params, mask, z_cond):
    h = mlp(params, z_cond * mask)
    d = mask.shape[0]
    shift, raw = h[:d], h[d:]
    log_scale = SCALE_BOUND * jnp.tanh(raw / SCALE_BOUND)
    return shift, log_scale


def coupling_forward(params: list[dict[str, Array]], mask: Array, z: Array) -> tuple[Array, Array]:
    """x = mask * z + (1 - mask) * (z * exp(s) + t) with (t, s) = conditioner(mask * z).

    The conditioner MLP outputs 2 * dim values: the first dim are the shift t, the second
    dim the raw log-scale, squashed as s = B tanh(raw / B) with B = SCALE_BOUND. Returns
    (x, log_det) with log_det = sum((1 - mask) * s)."""
    raise NotImplementedError  # Module 15a, Step 1


def coupling_inverse(params: list[dict[str, Array]], mask: Array, x: Array) -> tuple[Array, Array]:
    """Exact inverse of coupling_forward. Returns (z, log_det) where log_det is the
    log-determinant of the INVERSE map, i.e. -sum((1 - mask) * s)."""
    raise NotImplementedError  # Module 15a, Step 1


# --------------------------------------------------------------------------------------
# Step 2: the RealNVP stack with a final learned affine layer
# --------------------------------------------------------------------------------------
def init_flow(key: Array, dim: int, n_layers: int, hidden: int = 32) -> dict:
    """params = {"layers": [conditioner params per coupling], "act_log_scale": zeros(dim),
    "act_shift": zeros(dim)}. Each conditioner is init_conditioner with sizes
    dim -> hidden -> hidden -> 2 dim.
    The masks are not parameters; recompute them with make_masks(dim, n_layers)."""
    raise NotImplementedError  # Module 15a, Step 2


def flow_forward(params: dict, z: Array) -> tuple[Array, Array]:
    """Base -> target: couplings in order, then x = x * exp(act_log_scale) + act_shift.
    Returns (x, total log_det)."""
    raise NotImplementedError  # Module 15a, Step 2


def flow_inverse(params: dict, x: Array) -> tuple[Array, Array]:
    """Target -> base: undo the affine layer, then the couplings in reverse order.
    Returns (z, log_det of the inverse map)."""
    raise NotImplementedError  # Module 15a, Step 2


def _std_normal_log_prob(z: Array) -> Array:
    return -0.5 * jnp.sum(z**2) - 0.5 * z.shape[0] * jnp.log(2 * jnp.pi)


def flow_log_prob(params: dict, x: Array) -> Array:
    """log q(x) = log N(z; 0, I) + log|det J_{f^{-1}}(x)| with z = f^{-1}(x)."""
    raise NotImplementedError  # Module 15a, Step 2


def flow_sample(key: Array, params: dict, n: int) -> tuple[Array, Array]:
    """Draw n samples by pushing standard-normal noise through the flow.
    Returns (x [n, dim], log_q [n]) with log_q = log N(eps) - log|det J_f(eps)|."""
    raise NotImplementedError  # Module 15a, Step 2


# --------------------------------------------------------------------------------------
# Step 3: flow-based variational inference
# --------------------------------------------------------------------------------------
def flow_elbo(key: Array, params: dict, log_prob: LogProb, n_samples: int) -> Array:
    """Reparameterised ELBO: mean_s [ log p(f(eps_s)) + log|det J_f(eps_s)| ] + H[N(0, I)],
    with H[N(0, I)] = dim/2 log(2 pi e)."""
    raise NotImplementedError  # Module 15a, Step 3


def fit_flow_vi(log_prob: LogProb, params: dict, key: Array, n_steps: int, lr: float = 1e-3, n_samples: int = 16) -> tuple[dict, Array]:
    """Maximise flow_elbo with Adam (ascent, from Module 8) under lax.scan.
    Returns (params, elbo_trace [n_steps])."""
    raise NotImplementedError  # Module 15a, Step 3


# --------------------------------------------------------------------------------------
# Step 4: the funnel — reference posterior and tail statistics
# --------------------------------------------------------------------------------------
def hmc_reference(key: Array, n_warmup: int = 1000, n_samples: int = 1000) -> Array:
    """Reference posterior for the centred uplift model: run 4 adaptive HMC chains on
    the NON-centred parameterisation (Module 11 run_chains, q0 = 0.1 * N(0, I)), then map
    every draw to centred coordinates [mu, log tau, theta_1..8]. Returns [4 * n_samples, 10]."""
    raise NotImplementedError  # Module 15a, Step 4


def tail_mass(samples: Array, coord: int = 1, threshold: float = -1.0) -> Array:
    """Fraction of samples with samples[:, coord] < threshold (default: log tau < -1,
    the neck of the funnel)."""
    raise NotImplementedError  # Module 15a, Step 4


# --------------------------------------------------------------------------------------
# Step 5: importance-weighted bounds with the flow as proposal
# --------------------------------------------------------------------------------------
def importance_weighted_elbo(log_prob: LogProb, params: dict, key: Array, n_samples: int, k: int) -> Array:
    """IWAE-style bound L_k = mean over n_samples of [ logsumexp_j log w_j - log k ],
    w_j = p(x_j) / q(x_j) for k independent flow draws per outer sample.
    L_1 is the ELBO; L_k is non-decreasing in k and tends to log Z."""
    raise NotImplementedError  # Module 15a, Step 5


def importance_ess(log_w: Array) -> Array:
    """Effective sample size of normalised importance weights, (sum w)^2 / sum w^2,
    computed stably from log-weights."""
    raise NotImplementedError  # Module 15a, Step 5
