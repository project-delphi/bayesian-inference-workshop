"""Module 11 — MCMC diagnostics and the HMC-versus-VI comparison.

Split-R̂ and effective sample size following Gelman et al. (BDA3, Chapter 11) and
Vehtari et al. (2021), divergence counting from the energy error, multi-chain runs via
jax.vmap, and a quantitative comparison of an HMC posterior with a mean-field Gaussian
variational fit on the same SaaS-churn logistic regression.

Chains are always arrays of shape [m, n, d]: m chains, n draws, d coordinates.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .m09_hmc import HMCInfo, LogProb
from .m10_adaptation import Adapted, adaptive_hmc

Array = jax.Array


# --------------------------------------------------------------------------------------
# Step 1: split-R̂
# --------------------------------------------------------------------------------------
def split_rhat(chains: Array) -> Array:
    """Split each of the m chains in half (drop the last draw if n is odd), giving 2m
    chains of length n' = floor(n / 2). With chain means m_j, chain variances s_j^2
    (denominator n' - 1), W = mean_j s_j^2 and B = n' Var_j(m_j) (denominator 2m - 1),

        var_hat_plus = (n' - 1) / n' W + B / n',    R̂ = sqrt(var_hat_plus / W).

    Returns R̂ for each of the d coordinates."""
    raise NotImplementedError  # Module 11, Step 1


# --------------------------------------------------------------------------------------
# Step 2: autocorrelation and ESS
# --------------------------------------------------------------------------------------
def autocorrelation(x: Array) -> Array:
    """Normalised autocorrelation rho_t, t = 0..n-1, of a 1-D series via FFT.

    Zero-pad to at least 2n to avoid circular wrap-around, take |FFT|^2, inverse FFT,
    divide by the lag-0 value. rho_0 = 1."""
    raise NotImplementedError  # Module 11, Step 2


def ess(chains: Array) -> Array:
    """Effective sample size per coordinate, Vehtari et al. (2021) / Stan.

    For each coordinate: per-chain autocorrelations rho_{j,t} and variances s_j^2;
    W = mean_j s_j^2; var_hat_plus as in split_rhat (without splitting);
    combined rho_t = 1 - (W - mean_j s_j^2 rho_{j,t}) / var_hat_plus.
    Geyer's initial positive sequence: sum consecutive pairs P_k = rho_{2k} + rho_{2k+1}
    while P_k > 0 (and enforce monotone non-increasing P_k), then
    tau_hat = -1 + 2 sum_k P_k and ESS = m n / tau_hat. Cap ESS at m n log10(m n) as Stan does.
    Non-JAX NumPy is acceptable here (this function is a diagnostic, not sampled code).
    """
    raise NotImplementedError  # Module 11, Step 2


# --------------------------------------------------------------------------------------
# Step 3: divergences and multi-chain runs
# --------------------------------------------------------------------------------------
def divergences(infos: HMCInfo, threshold: float = 1000.0) -> Array:
    """Number of transitions whose energy error exceeds `threshold` (Stan's default),
    counting non-finite energy errors as divergent."""
    raise NotImplementedError  # Module 11, Step 3


def run_chains(log_prob: LogProb, key: Array, q0s: Array, n_warmup: int, n_samples: int, step_size0: float = 0.1, max_leapfrog: int = 16, target_accept: float = 0.8) -> tuple[Array, HMCInfo, Adapted]:
    """Run adaptive_hmc independently on each row of q0s [m, d] with jax.vmap over
    (key, q0). Returns (samples [m, n_samples, d], HMCInfo with leading axes [m, n],
    Adapted with leading axis m). Wrap in jax.jit with the integer arguments static."""
    raise NotImplementedError  # Module 11, Step 3


# --------------------------------------------------------------------------------------
# Step 4: summaries and the comparison with VI
# --------------------------------------------------------------------------------------
def summarize(chains: Array) -> dict[str, Array]:
    """Per-coordinate posterior summary from chains [m, n, d]:
    mean, sd, q05, q95, rhat (split_rhat), ess."""
    raise NotImplementedError  # Module 11, Step 4


def compare_hmc_vs_vi(hmc_chains: Array, vi_loc: Array, vi_scale: Array) -> dict[str, Array]:
    """Compare an HMC posterior (chains [m, n, d]) with a mean-field Gaussian q.
    Returns
      mean_diff_in_sd: (vi_loc - hmc_mean) / hmc_sd            [d]
      sd_ratio:        vi_scale / hmc_sd                         [d]
      max_abs_corr:    largest |off-diagonal| HMC posterior correlation (scalar);
                       mean-field VI assumes it is zero."""
    raise NotImplementedError  # Module 11, Step 4


# --------------------------------------------------------------------------------------
# The churn posterior and a Laplace reference
# --------------------------------------------------------------------------------------
def churn_log_joint(beta: Array, X: Array, y: Array, prior_scale: float = 2.5) -> Array:
    """Bayesian logistic regression: beta_k ~ N(0, prior_scale^2),
    y_i ~ Bernoulli(sigmoid(x_i . beta)). Same model as Module 7."""
    logits = X @ beta
    ll = jnp.sum(y * logits - jax.nn.softplus(logits))
    return ll + jnp.sum(-0.5 * (beta / prior_scale) ** 2)


def laplace_approximation(log_joint: LogProb, x0: Array, n_newton: int = 50) -> tuple[Array, Array]:
    """Mode by Newton's method and covariance = inverse negative Hessian at the mode."""
    g, H = jax.grad(log_joint), jax.hessian(log_joint)

    def step(_, x):
        return x - jnp.linalg.solve(H(x), g(x))

    mode = jax.lax.fori_loop(0, n_newton, step, x0)
    return mode, jnp.linalg.inv(-H(mode))
