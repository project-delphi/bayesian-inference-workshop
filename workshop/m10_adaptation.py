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
    """t = 0, h_bar = 0, log_step = log(step_size0), log_step_avg = log(step_size0),
    mu = log(10 step_size0). The first update gives the old average weight zero, so the
    initial log_step_avg only matters when no update follows: then the final step size
    is step_size0 rather than 1."""
    raise NotImplementedError  # Module 10, Step 1


def dual_averaging_update(state: DualAveragingState, accept_prob: Array, target: float = 0.8, gamma: float = 0.05, t0: float = 10.0, kappa: float = 0.75) -> DualAveragingState:
    """One Nesterov dual-averaging update (Hoffman & Gelman 2014, Algorithm 5):

        t <- t + 1
        h_bar <- (1 - 1/(t + t0)) h_bar + (target - accept_prob) / (t + t0)
        log_step <- mu - sqrt(t) / gamma * h_bar
        log_step_avg <- t^{-kappa} log_step + (1 - t^{-kappa}) log_step_avg
    """
    raise NotImplementedError  # Module 10, Step 1


def dual_averaging_final(state: DualAveragingState) -> Array:
    """The step size to use after warm-up: exp(log_step_avg)."""
    raise NotImplementedError  # Module 10, Step 1


# --------------------------------------------------------------------------------------
# Step 2: Welford's online variance for the diagonal mass matrix
# --------------------------------------------------------------------------------------
class WelfordState(NamedTuple):
    n: Array
    mean: Array
    m2: Array


def welford_init(dim: int) -> WelfordState:
    raise NotImplementedError  # Module 10, Step 2


def welford_update(state: WelfordState, x: Array) -> WelfordState:
    """n <- n + 1; delta = x - mean; mean <- mean + delta / n; m2 <- m2 + delta (x - mean)."""
    raise NotImplementedError  # Module 10, Step 2


def welford_finalize(state: WelfordState, regularize: bool = True) -> Array:
    """Sample variance m2 / (n - 1). With `regularize`, shrink toward the identity as
    Stan does: var <- n/(n+5) var + 1e-3 * 5/(n+5). Returns the diagonal inverse mass."""
    raise NotImplementedError  # Module 10, Step 2


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
    raise NotImplementedError  # Module 10, Step 3


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
    raise NotImplementedError  # Module 10, Step 3


# --------------------------------------------------------------------------------------
# Step 4: adaptive HMC with jittered trajectory length
# --------------------------------------------------------------------------------------
def hmc_jittered(log_prob: LogProb, key: Array, q0: Array, n_samples: int, step_size: Array, max_leapfrog: int, inv_mass: Array) -> tuple[Array, HMCInfo]:
    """Fixed step size and mass, but n_leapfrog ~ Uniform{1..max_leapfrog} each
    iteration (breaks the resonance of fixed-length trajectories on near-Gaussian
    targets). Returns (samples [n, d], HMCInfo)."""
    raise NotImplementedError  # Module 10, Step 4


def adaptive_hmc(log_prob: LogProb, key: Array, q0: Array, n_warmup: int, n_samples: int, step_size0: float = 0.1, max_leapfrog: int = 16, target_accept: float = 0.8) -> tuple[Array, HMCInfo, Adapted]:
    """warmup(...) followed by hmc_jittered(...) from the last warm-up state.
    Returns (samples [n_samples, d], sampling-phase HMCInfo, Adapted)."""
    raise NotImplementedError  # Module 10, Step 4


# --------------------------------------------------------------------------------------
# Step 5: the U-turn criterion
# --------------------------------------------------------------------------------------
def uturn(q_minus: Array, q_plus: Array, p_minus: Array, p_plus: Array, inv_mass: Array) -> Array:
    """NUTS termination criterion for a trajectory with endpoints (q-, p-) and (q+, p+):
    True when (q+ - q-) . (M^{-1} p-) < 0  or  (q+ - q-) . (M^{-1} p+) < 0, i.e. when
    continuing in either direction would shrink the distance between the endpoints."""
    raise NotImplementedError  # Module 10, Step 5


def trajectory_length_study(log_prob: LogProb, q0: Array, p0: Array, step_size: float, max_steps: int, inv_mass: Array) -> tuple[Array, Array]:
    """Integrate forward from (q0, p0) one leapfrog step at a time and record, for each
    step count L = 1..max_steps, whether the U-turn criterion between (q0, p0) and the
    current point fires. Returns (uturn_flags [max_steps], first_uturn_step) where
    first_uturn_step is the smallest L with a U-turn (max_steps if none)."""
    raise NotImplementedError  # Module 10, Step 5
