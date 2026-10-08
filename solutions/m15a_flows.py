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
    # [m15a step 1]
    params = init_mlp(key, sizes)
    params[-1] = {"w": jnp.zeros_like(params[-1]["w"]), "b": params[-1]["b"]}
    return params


def make_masks(dim: int, n_layers: int) -> Array:
    """Alternating binary masks [n_layers, dim]: layer k conditions on the coordinates
    with (index % 2 == k % 2) and transforms the others."""
    # [m15a step 1]
    idx = jnp.arange(dim)
    return jnp.stack([(idx % 2 == k % 2).astype(float) for k in range(n_layers)])


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
    # [m15a step 1]
    shift, log_scale = _shift_log_scale(params, mask, z)
    x = mask * z + (1 - mask) * (z * jnp.exp(log_scale) + shift)
    return x, jnp.sum((1 - mask) * log_scale)


def coupling_inverse(params: list[dict[str, Array]], mask: Array, x: Array) -> tuple[Array, Array]:
    """Exact inverse of coupling_forward. Returns (z, log_det) where log_det is the
    log-determinant of the INVERSE map, i.e. -sum((1 - mask) * s)."""
    # [m15a step 1]
    shift, log_scale = _shift_log_scale(params, mask, x)
    z = mask * x + (1 - mask) * ((x - shift) * jnp.exp(-log_scale))
    return z, -jnp.sum((1 - mask) * log_scale)


# --------------------------------------------------------------------------------------
# Step 2: the RealNVP stack with a final learned affine layer
# --------------------------------------------------------------------------------------
def init_flow(key: Array, dim: int, n_layers: int, hidden: int = 32) -> dict:
    """params = {"layers": [conditioner params per coupling], "act_log_scale": zeros(dim),
    "act_shift": zeros(dim)}. Each conditioner is init_conditioner with sizes
    dim -> hidden -> hidden -> 2 dim.
    The masks are not parameters; recompute them with make_masks(dim, n_layers)."""
    # [m15a step 2]
    keys = jax.random.split(key, n_layers)
    layers = [init_conditioner(k, [dim, hidden, hidden, 2 * dim]) for k in keys]
    return {"layers": layers, "act_log_scale": jnp.zeros(dim), "act_shift": jnp.zeros(dim)}


def flow_forward(params: dict, z: Array) -> tuple[Array, Array]:
    """Base -> target: couplings in order, then x = x * exp(act_log_scale) + act_shift.
    Returns (x, total log_det)."""
    # [m15a step 2]
    masks = make_masks(z.shape[0], len(params["layers"]))
    log_det = jnp.zeros(())
    x = z
    for layer, mask in zip(params["layers"], masks):
        x, ld = coupling_forward(layer, mask, x)
        log_det = log_det + ld
    x = x * jnp.exp(params["act_log_scale"]) + params["act_shift"]
    return x, log_det + jnp.sum(params["act_log_scale"])


def flow_inverse(params: dict, x: Array) -> tuple[Array, Array]:
    """Target -> base: undo the affine layer, then the couplings in reverse order.
    Returns (z, log_det of the inverse map)."""
    # [m15a step 2]
    masks = make_masks(x.shape[0], len(params["layers"]))
    z = (x - params["act_shift"]) * jnp.exp(-params["act_log_scale"])
    log_det = -jnp.sum(params["act_log_scale"])
    for layer, mask in zip(reversed(params["layers"]), reversed(list(masks))):
        z, ld = coupling_inverse(layer, mask, z)
        log_det = log_det + ld
    return z, log_det


def _std_normal_log_prob(z: Array) -> Array:
    return -0.5 * jnp.sum(z**2) - 0.5 * z.shape[0] * jnp.log(2 * jnp.pi)


def flow_log_prob(params: dict, x: Array) -> Array:
    """log q(x) = log N(z; 0, I) + log|det J_{f^{-1}}(x)| with z = f^{-1}(x)."""
    # [m15a step 2]
    z, log_det = flow_inverse(params, x)
    return _std_normal_log_prob(z) + log_det


def flow_sample(key: Array, params: dict, n: int) -> tuple[Array, Array]:
    """Draw n samples by pushing standard-normal noise through the flow.
    Returns (x [n, dim], log_q [n]) with log_q = log N(eps) - log|det J_f(eps)|."""
    # [m15a step 2]
    dim = params["act_shift"].shape[0]
    eps = jax.random.normal(key, (n, dim))
    x, log_det = jax.vmap(lambda e: flow_forward(params, e))(eps)
    return x, jax.vmap(_std_normal_log_prob)(eps) - log_det


# --------------------------------------------------------------------------------------
# Step 3: flow-based variational inference
# --------------------------------------------------------------------------------------
def flow_elbo(key: Array, params: dict, log_prob: LogProb, n_samples: int) -> Array:
    """Reparameterised ELBO: mean_s [ log p(f(eps_s)) + log|det J_f(eps_s)| ] + H[N(0, I)],
    with H[N(0, I)] = dim/2 log(2 pi e)."""
    # [m15a step 3]
    dim = params["act_shift"].shape[0]
    eps = jax.random.normal(key, (n_samples, dim))
    x, log_det = jax.vmap(lambda e: flow_forward(params, e))(eps)
    return jnp.mean(jax.vmap(log_prob)(x) + log_det) + 0.5 * dim * jnp.log(2 * jnp.pi * jnp.e)


def fit_flow_vi(log_prob: LogProb, params: dict, key: Array, n_steps: int, lr: float = 1e-3, n_samples: int = 16) -> tuple[dict, Array]:
    """Maximise flow_elbo with Adam (ascent, from Module 8) under lax.scan.
    Returns (params, elbo_trace [n_steps])."""
    # [m15a step 3]
    state = adam_init(params)
    keys = jax.random.split(key, n_steps)

    def body(carry, k):
        p, s = carry
        value, grads = jax.value_and_grad(flow_elbo, argnums=1)(k, p, log_prob, n_samples)
        p, s = adam_update(grads, s, p, lr)
        return (p, s), value

    (params, _), trace = lax.scan(body, (params, state), keys)
    return params, trace


# --------------------------------------------------------------------------------------
# Step 4: the funnel — reference posterior and tail statistics
# --------------------------------------------------------------------------------------
def hmc_reference(key: Array, n_warmup: int = 1000, n_samples: int = 1000) -> Array:
    """Reference posterior for the centred uplift model: run 4 adaptive HMC chains on
    the NON-centred parameterisation (Module 11 run_chains, q0 = 0.1 * N(0, I)), then map
    every draw to centred coordinates [mu, log tau, theta_1..8]. Returns [4 * n_samples, 10]."""
    # [m15a step 4]
    k_init, k_run = jax.random.split(key)
    q0s = 0.1 * jax.random.normal(k_init, (4, UPLIFT_DIM))
    chains, _, _ = run_chains(uplift_noncentered_log_prob, k_run, q0s, n_warmup, n_samples)
    return uplift_noncentered_to_centered(chains).reshape(-1, UPLIFT_DIM)


def tail_mass(samples: Array, coord: int = 1, threshold: float = -1.0) -> Array:
    """Fraction of samples with samples[:, coord] < threshold (default: log tau < -1,
    the neck of the funnel)."""
    # [m15a step 4]
    return jnp.mean(samples[:, coord] < threshold)


# --------------------------------------------------------------------------------------
# Step 5: importance-weighted bounds with the flow as proposal
# --------------------------------------------------------------------------------------
def importance_weighted_elbo(log_prob: LogProb, params: dict, key: Array, n_samples: int, k: int) -> Array:
    """IWAE-style bound L_k = mean over n_samples of [ logsumexp_j log w_j - log k ],
    w_j = p(x_j) / q(x_j) for k independent flow draws per outer sample.
    L_1 is the ELBO; L_k is non-decreasing in k and tends to log Z."""
    # [m15a step 5]
    x, log_q = flow_sample(key, params, n_samples * k)
    log_w = (jax.vmap(log_prob)(x) - log_q).reshape(n_samples, k)
    return jnp.mean(jax.nn.logsumexp(log_w, axis=1) - jnp.log(k))


def importance_ess(log_w: Array) -> Array:
    """Effective sample size of normalised importance weights, (sum w)^2 / sum w^2,
    computed stably from log-weights."""
    # [m15a step 5]
    lw = log_w - jnp.max(log_w)
    return jnp.exp(2 * jax.nn.logsumexp(lw) - jax.nn.logsumexp(2 * lw))
