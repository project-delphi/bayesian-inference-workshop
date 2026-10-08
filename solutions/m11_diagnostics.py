"""Module 11 — MCMC diagnostics and the HMC-versus-VI comparison.

Split-R̂ and effective sample size following Gelman et al. (BDA3, Chapter 11) and
Vehtari et al. (2021), divergence counting from the energy error, multi-chain runs via
jax.vmap, and a quantitative comparison of an HMC posterior with a mean-field Gaussian
variational fit on the same SaaS-churn logistic regression. The model and its Laplace
reference are Module 7's `churn_log_joint` and `laplace`.

Chains are always arrays of shape [m, n, d]: m chains, n draws, d coordinates.
"""
from __future__ import annotations

from functools import partial
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
    # [m11 step 1]
    m, n, d = chains.shape
    half = n // 2
    split = jnp.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)  # [2m, half, d]
    means = split.mean(axis=1)  # [2m, d]
    vars_ = split.var(axis=1, ddof=1)  # [2m, d]
    W = vars_.mean(axis=0)
    B = half * means.var(axis=0, ddof=1)
    var_plus = (half - 1) / half * W + B / half
    return jnp.sqrt(var_plus / W)


# --------------------------------------------------------------------------------------
# Step 2: autocorrelation and ESS
# --------------------------------------------------------------------------------------
def autocorrelation(x: Array) -> Array:
    """Normalised autocorrelation rho_t, t = 0..n-1, of a 1-D series via FFT.

    Zero-pad to at least 2n to avoid circular wrap-around, take |FFT|^2, inverse FFT,
    divide by the lag-0 value. rho_0 = 1."""
    # [m11 step 2]
    n = x.shape[0]
    xc = x - x.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = jnp.fft.rfft(xc, n=size)
    acov = jnp.fft.irfft(f * jnp.conj(f), n=size)[:n]
    return acov / acov[0]


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
    # [m11 step 2]
    chains = np.asarray(chains)
    m, n, d = chains.shape
    out = np.zeros(d)
    for k in range(d):
        x = chains[:, :, k]
        s2 = x.var(axis=1, ddof=1)  # [m]
        W = s2.mean()
        if m > 1:
            B = n * x.mean(axis=1).var(ddof=1)
        else:
            B = 0.0
        var_plus = (n - 1) / n * W + B / n
        rho_j = np.stack([np.asarray(autocorrelation(jnp.asarray(x[j]))) for j in range(m)])  # [m, n]
        rho = 1.0 - (W - (s2[:, None] * rho_j).mean(axis=0)) / var_plus  # [n]
        # Geyer initial positive sequence on pair sums
        max_pairs = n // 2
        pairs = rho[: 2 * max_pairs].reshape(max_pairs, 2).sum(axis=1)
        tau = 0.0
        prev = np.inf
        for P in pairs:
            if P <= 0:
                break
            P = min(P, prev)  # initial monotone sequence
            prev = P
            tau += P
        tau_hat = -1.0 + 2.0 * tau
        tau_hat = max(tau_hat, 1e-12)
        e = m * n / tau_hat
        out[k] = min(e, m * n * np.log10(max(m * n, 10)))
    return jnp.asarray(out)


# --------------------------------------------------------------------------------------
# Step 3: divergences and multi-chain runs
# --------------------------------------------------------------------------------------
def divergences(infos: HMCInfo, threshold: float = 1000.0) -> Array:
    """Number of transitions whose energy error exceeds `threshold` (Stan's default),
    counting non-finite energy errors as divergent."""
    # [m11 step 3]
    e = infos.energy_error
    return jnp.sum(~jnp.isfinite(e) | (e > threshold))


@partial(jax.jit, static_argnames=("log_prob", "n_warmup", "n_samples", "max_leapfrog"))
def run_chains(log_prob: LogProb, key: Array, q0s: Array, n_warmup: int, n_samples: int, step_size0: float = 0.1, max_leapfrog: int = 16, target_accept: float = 0.8) -> tuple[Array, HMCInfo, Adapted]:
    """Run adaptive_hmc independently on each row of q0s [m, d] with jax.vmap over
    (key, q0), one key per chain from jax.random.split. Returns (samples
    [m, n_samples, d], HMCInfo with leading axes [m, n], Adapted with leading axis m).

    The decorator compiles one program per target function and per value of the
    integer arguments; a second call with the same `log_prob` object reuses it. A
    lambda built afresh for each call is a new object and compiles again."""
    # [m11 step 3]
    keys = jax.random.split(key, q0s.shape[0])
    return jax.vmap(lambda k, q: adaptive_hmc(log_prob, k, q, n_warmup, n_samples, step_size0, max_leapfrog, target_accept))(keys, q0s)


# --------------------------------------------------------------------------------------
# Step 4: summaries and the comparison with VI
# --------------------------------------------------------------------------------------
def summarize(chains: Array) -> dict[str, Array]:
    """Per-coordinate posterior summary from chains [m, n, d]:
    mean, sd, q05, q95, rhat (split_rhat), ess."""
    # [m11 step 4]
    flat = chains.reshape(-1, chains.shape[-1])
    return {
        "mean": flat.mean(0),
        "sd": flat.std(0, ddof=1),
        "q05": jnp.quantile(flat, 0.05, axis=0),
        "q95": jnp.quantile(flat, 0.95, axis=0),
        "rhat": split_rhat(chains),
        "ess": ess(chains),
    }


def compare_hmc_vs_vi(hmc_chains: Array, vi_loc: Array, vi_scale: Array) -> dict[str, Array]:
    """Compare an HMC posterior (chains [m, n, d]) with a mean-field Gaussian q.
    Returns
      mean_diff_in_sd: (vi_loc - hmc_mean) / hmc_sd            [d]
      sd_ratio:        vi_scale / hmc_sd                         [d]
      max_abs_corr:    largest |off-diagonal| HMC posterior correlation (scalar);
                       mean-field VI assumes it is zero."""
    # [m11 step 4]
    flat = hmc_chains.reshape(-1, hmc_chains.shape[-1])
    mean, sd = flat.mean(0), flat.std(0, ddof=1)
    corr = jnp.corrcoef(flat.T)
    off = corr - jnp.diag(jnp.diag(corr))
    return {
        "mean_diff_in_sd": (vi_loc - mean) / sd,
        "sd_ratio": vi_scale / sd,
        "max_abs_corr": jnp.max(jnp.abs(off)),
    }

