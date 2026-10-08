"""Module 10 — Adaptation: step size, mass matrix, trajectory length.

Dual averaging (Hoffman & Gelman 2014, Algorithm 5) tunes the step size toward a target
acceptance statistic; Welford's online variance estimates a diagonal mass matrix in
slow adaptation windows; a Stan-style windowed warm-up schedule combines the two; and
the trajectory length is jittered to break resonances. The U-turn criterion of NUTS is
implemented on its own; the full tree-building algorithm is the Challenge.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .m09_hmc import HMCInfo, LogProb, hmc_step, leapfrog

Array = jax.Array


# --------------------------------------------------------------------------------------
# Step 1: dual averaging for the step size
# --------------------------------------------------------------------------------------
class DualAveragingState(NamedTuple):
    log_step: Array  # current log step size (used for the next transition)
    log_step_avg: Array  # running weighted average of log step (used after warm-up)
    h_bar: Array  # running average of (target - accept_prob)
    t: Array  # iteration counter (float)
    mu: Array  # shrinkage point, log(10 * step_size0)


def dual_averaging_init(step_size0: float) -> DualAveragingState:
    """t = 0, h_bar = 0, log_step = log(step_size0), log_step_avg = 0, mu = log(10 step_size0)."""
    # [m10 step 1]
    s0 = jnp.asarray(step_size0, dtype=float)
    return DualAveragingState(
        log_step=jnp.log(s0),
        log_step_avg=jnp.asarray(0.0),
        h_bar=jnp.asarray(0.0),
        t=jnp.asarray(0.0),
        mu=jnp.log(10.0 * s0),
    )


def dual_averaging_update(state: DualAveragingState, accept_prob: Array, target: float = 0.8, gamma: float = 0.05, t0: float = 10.0, kappa: float = 0.75) -> DualAveragingState:
    """One Nesterov dual-averaging update (Hoffman & Gelman 2014, Algorithm 5):

        t <- t + 1
        h_bar <- (1 - 1/(t + t0)) h_bar + (target - accept_prob) / (t + t0)
        log_step <- mu - sqrt(t) / gamma * h_bar
        log_step_avg <- t^{-kappa} log_step + (1 - t^{-kappa}) log_step_avg
    """
    # [m10 step 1]
    t = state.t + 1.0
    eta_h = 1.0 / (t + t0)
    h_bar = (1.0 - eta_h) * state.h_bar + eta_h * (target - accept_prob)
    log_step = state.mu - jnp.sqrt(t) / gamma * h_bar
    w = t ** (-kappa)
    log_step_avg = w * log_step + (1.0 - w) * state.log_step_avg
    return DualAveragingState(log_step=log_step, log_step_avg=log_step_avg, h_bar=h_bar, t=t, mu=state.mu)


def dual_averaging_final(state: DualAveragingState) -> Array:
    """The step size to use after warm-up: exp(log_step_avg)."""
    # [m10 step 1]
    return jnp.exp(state.log_step_avg)


# --------------------------------------------------------------------------------------
# Step 2: Welford's online variance for the diagonal mass matrix
# --------------------------------------------------------------------------------------
class WelfordState(NamedTuple):
    n: Array
    mean: Array
    m2: Array


def welford_init(dim: int) -> WelfordState:
    # [m10 step 2]
    return WelfordState(n=jnp.asarray(0.0), mean=jnp.zeros(dim), m2=jnp.zeros(dim))


def welford_update(state: WelfordState, x: Array) -> WelfordState:
    """n <- n + 1; delta = x - mean; mean <- mean + delta / n; m2 <- m2 + delta (x - mean)."""
    # [m10 step 2]
    n = state.n + 1.0
    delta = x - state.mean
    mean = state.mean + delta / n
    m2 = state.m2 + delta * (x - mean)
    return WelfordState(n=n, mean=mean, m2=m2)


def welford_finalize(state: WelfordState, regularize: bool = True) -> Array:
    """Sample variance m2 / (n - 1). With `regularize`, shrink toward the identity as
    Stan does: var <- n/(n+5) var + 1e-3 * 5/(n+5). Returns the diagonal inverse mass."""
    # [m10 step 2]
    n = state.n
    var = state.m2 / jnp.maximum(n - 1.0, 1.0)
    if regularize:
        var = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
    return var


# --------------------------------------------------------------------------------------
# Step 3: windowed warm-up
# --------------------------------------------------------------------------------------
def warmup_schedule(n_warmup: int, init_frac: float = 0.15, term_frac: float = 0.10, base_window: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Stan-style schedule. Returns two boolean arrays of length n_warmup:
    `is_slow[i]` — iteration i belongs to a slow (mass-matrix) window;
    `window_end[i]` — iteration i closes a slow window (update the mass matrix, reset
    dual averaging and the Welford accumulator after this iteration).

    Layout: an initial fast window of ceil(init_frac * n), slow windows of sizes
    base_window, 2 base_window, 4 base_window, ... (the last one extended to fill the
    space), and a terminal fast window of ceil(term_frac * n). If n_warmup < 20 there
    are no slow windows at all.
    """
    # [m10 step 3]
    is_slow = np.zeros(n_warmup, dtype=bool)
    window_end = np.zeros(n_warmup, dtype=bool)
    if n_warmup < 20:
        return is_slow, window_end
    init = int(np.ceil(init_frac * n_warmup))
    term = int(np.ceil(term_frac * n_warmup))
    slow_start, slow_end = init, n_warmup - term  # [slow_start, slow_end)
    if slow_end - slow_start < base_window:
        return is_slow, window_end
    start = slow_start
    size = base_window
    while start < slow_end:
        end = start + size
        if end + 2 * size > slow_end:  # not enough room for the next window: extend
            end = slow_end
        is_slow[start:end] = True
        window_end[end - 1] = True
        start = end
        size *= 2
    return is_slow, window_end


class Adapted(NamedTuple):
    step_size: Array
    inv_mass: Array
    q: Array  # last warm-up state


def warmup(log_prob: LogProb, key: Array, q0: Array, n_warmup: int, step_size0: float = 0.1, max_leapfrog: int = 16, target_accept: float = 0.8) -> tuple[Adapted, HMCInfo]:
    """Windowed warm-up. Each iteration: draw n_leapfrog ~ Uniform{1..max_leapfrog},
    take one HMC step at the current step size and inverse mass, update dual averaging
    with the acceptance probability; in slow windows feed the state to Welford; at a
    window end replace inv_mass by the Welford variance, reset Welford and restart dual
    averaging from the current step size. Final step size is dual_averaging_final.

    Returns (Adapted(step_size, inv_mass, q), per-iteration HMCInfo). Implement with
    lax.scan over the precomputed schedule arrays.
    """
    # [m10 step 3]
    is_slow, window_end = (jnp.asarray(a) for a in warmup_schedule(n_warmup))
    dim = q0.shape[0]

    def body(carry, sched):
        q, da, wf, inv_mass, key = carry
        slow, end = sched
        key, k_l, k_h = jax.random.split(key, 3)
        n_leap = jax.random.randint(k_l, (), 1, max_leapfrog + 1)
        q, info = hmc_step(log_prob, k_h, q, jnp.exp(da.log_step), n_leap, inv_mass)
        da = dual_averaging_update(da, info.accept_prob, target=target_accept)
        wf_upd = welford_update(wf, q)
        wf = jax.tree_util.tree_map(lambda a, b: jnp.where(slow, a, b), wf_upd, wf)
        # window end: new mass matrix, fresh accumulators, dual averaging restarted
        new_inv_mass = welford_finalize(wf)
        inv_mass = jnp.where(end, new_inv_mass, inv_mass)
        wf = jax.tree_util.tree_map(lambda a, b: jnp.where(end, a, b), welford_init(dim), wf)
        da_reset = dual_averaging_init(jnp.exp(da.log_step))
        da = jax.tree_util.tree_map(lambda a, b: jnp.where(end, a, b), da_reset, da)
        return (q, da, wf, inv_mass, key), info

    init = (q0, dual_averaging_init(step_size0), welford_init(dim), jnp.ones(dim), key)
    (q, da, _, inv_mass, _), infos = lax.scan(body, init, (is_slow, window_end))
    return Adapted(step_size=dual_averaging_final(da), inv_mass=inv_mass, q=q), infos


# --------------------------------------------------------------------------------------
# Step 4: adaptive HMC with jittered trajectory length
# --------------------------------------------------------------------------------------
def hmc_jittered(log_prob: LogProb, key: Array, q0: Array, n_samples: int, step_size: Array, max_leapfrog: int, inv_mass: Array) -> tuple[Array, HMCInfo]:
    """Fixed step size and mass, but n_leapfrog ~ Uniform{1..max_leapfrog} each
    iteration (breaks the resonance of fixed-length trajectories on near-Gaussian
    targets). Returns (samples [n, d], HMCInfo)."""
    # [m10 step 4]

    def body(carry, _):
        q, key = carry
        key, k_l, k_h = jax.random.split(key, 3)
        n_leap = jax.random.randint(k_l, (), 1, max_leapfrog + 1)
        q, info = hmc_step(log_prob, k_h, q, step_size, n_leap, inv_mass)
        return (q, key), (q, info)

    (_, _), (qs, infos) = lax.scan(body, (q0, key), None, length=n_samples)
    return qs, infos


def adaptive_hmc(log_prob: LogProb, key: Array, q0: Array, n_warmup: int, n_samples: int, step_size0: float = 0.1, max_leapfrog: int = 16, target_accept: float = 0.8) -> tuple[Array, HMCInfo, Adapted]:
    """warmup(...) followed by hmc_jittered(...) from the last warm-up state.
    Returns (samples [n_samples, d], sampling-phase HMCInfo, Adapted)."""
    # [m10 step 4]
    k_w, k_s = jax.random.split(key)
    adapted, _ = warmup(log_prob, k_w, q0, n_warmup, step_size0, max_leapfrog, target_accept)
    samples, infos = hmc_jittered(log_prob, k_s, adapted.q, n_samples, adapted.step_size, max_leapfrog, adapted.inv_mass)
    return samples, infos, adapted


# --------------------------------------------------------------------------------------
# Step 5: the U-turn criterion
# --------------------------------------------------------------------------------------
def uturn(q_minus: Array, q_plus: Array, p_minus: Array, p_plus: Array, inv_mass: Array) -> Array:
    """NUTS termination criterion for a trajectory with endpoints (q-, p-) and (q+, p+):
    True when (q+ - q-) . (M^{-1} p-) < 0  or  (q+ - q-) . (M^{-1} p+) < 0, i.e. when
    continuing in either direction would shrink the distance between the endpoints."""
    # [m10 step 5]
    dq = q_plus - q_minus
    return (jnp.dot(dq, inv_mass * p_minus) < 0.0) | (jnp.dot(dq, inv_mass * p_plus) < 0.0)


def trajectory_length_study(log_prob: LogProb, q0: Array, p0: Array, step_size: float, max_steps: int, inv_mass: Array) -> tuple[Array, Array]:
    """Integrate forward from (q0, p0) one leapfrog step at a time and record, for each
    step count L = 1..max_steps, whether the U-turn criterion between (q0, p0) and the
    current point fires. Returns (uturn_flags [max_steps], first_uturn_step) where
    first_uturn_step is the smallest L with a U-turn (max_steps if none)."""
    # [m10 step 5]
    grad = jax.grad(log_prob)

    def body(qp, _):
        q, p = leapfrog(grad, qp[0], qp[1], step_size, 1, inv_mass)
        return (q, p), uturn(q0, q, p0, p, inv_mass)

    _, flags = lax.scan(body, (q0, p0), None, length=max_steps)
    first = jnp.where(jnp.any(flags), jnp.argmax(flags) + 1, max_steps)
    return flags, first
