"""Deterministic synthetic datasets used throughout the workshop.

Every generator takes an explicit integer seed and returns NumPy arrays (float64 /
int64). Nothing is downloaded. Each dataset is themed on a real problem class so model
parameters have a concrete reading:

- A/B conversion counts (marketing)               -> Beta-Bernoulli, hierarchical models
- Marketing channel attribution counts            -> Dirichlet-Categorical
- qPCR log-expression replicates (molecular bio)  -> Normal-Gamma
- Single-cell 2-D embedding with cell types       -> Gaussian mixture / CAVI
- SaaS churn table                                 -> Bayesian logistic regression
- Ad click-through table, 50k rows                 -> stochastic VI with subsampling
- Regional uplift estimates (eight-schools form)   -> hierarchical model, HMC funnel
- Michaelis-Menten enzyme kinetics                 -> non-linear regression in the PPL
- Transcriptional-program binary matrix            -> VAE
- Ramachandran-style torsion angles                -> diffusion model
- Ad-spend structural causal model                 -> counterfactuals
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------------------
# Marketing: A/B test
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ABTest:
    """Binary conversion outcomes for two landing-page variants."""

    a: np.ndarray  # shape (n_a,), 0/1
    b: np.ndarray  # shape (n_b,), 0/1
    true_rate_a: float
    true_rate_b: float


def ab_test_conversions(seed: int = 0, n_a: int = 400, n_b: int = 380) -> ABTest:
    rng = _rng(seed)
    pa, pb = 0.062, 0.079
    return ABTest(
        a=rng.binomial(1, pa, size=n_a).astype(np.int64),
        b=rng.binomial(1, pb, size=n_b).astype(np.int64),
        true_rate_a=pa,
        true_rate_b=pb,
    )


# --------------------------------------------------------------------------------------
# Marketing: channel attribution
# --------------------------------------------------------------------------------------
CHANNELS = ("organic_search", "paid_search", "social", "email", "referral", "direct")


def channel_attribution(seed: int = 0, n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Last-touch attribution channel for n conversions.

    Returns (labels in {0..5}, true_probs)."""
    rng = _rng(seed)
    p = np.array([0.31, 0.22, 0.14, 0.12, 0.09, 0.12])
    labels = rng.choice(len(CHANNELS), size=n, p=p).astype(np.int64)
    return labels, p


# --------------------------------------------------------------------------------------
# Molecular biology: qPCR replicates
# --------------------------------------------------------------------------------------
def qpcr_log_expression(seed: int = 0, n: int = 12) -> tuple[np.ndarray, float, float]:
    """Log2 fold-change replicates for one gene from a qPCR assay.

    Returns (samples, true_mean, true_precision)."""
    rng = _rng(seed)
    mu, prec = 1.8, 1 / 0.35**2
    x = rng.normal(mu, prec**-0.5, size=n)
    return x, mu, prec


# --------------------------------------------------------------------------------------
# Molecular biology: single-cell embedding
# --------------------------------------------------------------------------------------
CELL_TYPES = ("T cell", "B cell", "monocyte", "NK cell")


def single_cell_embedding(seed: int = 0, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2-D embedding of n cells drawn from 4 cell types.

    Returns (x [n,2], labels [n], true_means [4,2], true_weights [4]). Shared isotropic
    variance 0.35^2 so the known-variance GMM of Module 6 is well specified."""
    rng = _rng(seed)
    means = np.array([[-2.5, 0.0], [2.0, 2.0], [2.5, -2.0], [0.0, -3.0]])
    w = np.array([0.40, 0.25, 0.20, 0.15])
    z = rng.choice(4, size=n, p=w)
    x = means[z] + 0.35 * rng.normal(size=(n, 2))
    return x, z.astype(np.int64), means, w


# --------------------------------------------------------------------------------------
# Tech: SaaS churn
# --------------------------------------------------------------------------------------
CHURN_FEATURES = ("intercept", "tenure_months_std", "weekly_active_days_std", "support_tickets_std", "enterprise_plan")


def saas_churn(seed: int = 0, n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Churn-within-90-days labels for n SaaS accounts.

    Returns (X [n,5] with leading intercept column, y [n] in {0,1}, true_beta [5])."""
    rng = _rng(seed)
    tenure = rng.gamma(2.0, 9.0, size=n)
    active = np.clip(rng.normal(3.2, 1.6, size=n), 0, 7)
    tickets = rng.poisson(1.3, size=n).astype(float)
    enterprise = rng.binomial(1, 0.3, size=n).astype(float)
    z = lambda v: (v - v.mean()) / v.std()
    X = np.column_stack([np.ones(n), z(tenure), z(active), z(tickets), enterprise])
    beta = np.array([-1.1, -0.8, -1.2, 0.7, -0.5])
    p = 1 / (1 + np.exp(-X @ beta))
    y = rng.binomial(1, p).astype(np.int64)
    return X, y, beta


# --------------------------------------------------------------------------------------
# Tech: ad click-through, large
# --------------------------------------------------------------------------------------
def ad_clicks(seed: int = 0, n: int = 50_000, d: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Click labels for n ad impressions with d standardised features plus intercept.

    Returns (X [n,d+1], y [n], true_beta [d+1]). Base CTR about 4%."""
    rng = _rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d))])
    beta = np.concatenate([[-3.2], rng.normal(0, 0.5, size=d)])
    p = 1 / (1 + np.exp(-X @ beta))
    y = rng.binomial(1, p).astype(np.int64)
    return X, y, beta


# --------------------------------------------------------------------------------------
# Marketing: regional uplift (eight-schools structure)
# --------------------------------------------------------------------------------------
REGIONS = ("US-East", "US-West", "UK", "DE", "FR", "JP", "BR", "IN")


def regional_uplift(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Estimated conversion-rate uplift (percentage points) and standard errors from
    eight regional A/B tests of the same feature. Returns (y [8], sigma [8])."""
    rng = _rng(seed)
    sigma = np.array([1.5, 1.0, 1.6, 1.1, 0.9, 1.8, 2.0, 1.3])
    mu, tau = 0.8, 0.6
    theta = rng.normal(mu, tau, size=8)
    y = rng.normal(theta, sigma)
    return y, sigma


# --------------------------------------------------------------------------------------
# Molecular biology: Michaelis-Menten kinetics
# --------------------------------------------------------------------------------------
def enzyme_kinetics(seed: int = 0, n: int = 24) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Initial reaction velocities at n substrate concentrations.

    v = Vmax * s / (Km + s) + noise. Returns (s, v, Vmax, Km, noise_sd)."""
    rng = _rng(seed)
    vmax, km, sd = 12.0, 2.5, 0.6
    s = np.repeat(np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]), n // 8)
    v = vmax * s / (km + s) + sd * rng.normal(size=s.shape)
    return s, v, vmax, km, sd


# --------------------------------------------------------------------------------------
# Molecular biology: transcriptional programs (binary 8x8 gene grids)
# --------------------------------------------------------------------------------------
def gene_programs(seed: int = 0, n: int = 2000, k: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary presence/absence of 64 genes across n cells.

    Each cell activates a sparse subset of k latent programs; each program switches on a
    fixed block of genes. Returns (X [n,64] in {0,1}, program_indicators [n,k],
    program_gene_masks [k,64])."""
    rng = _rng(seed)
    masks = np.zeros((k, 64))
    # Programs correspond to rows/columns of the 8x8 grid, so samples are visualisable.
    for j in range(k):
        grid = np.zeros((8, 8))
        if j % 2 == 0:
            grid[(j // 2) * 2 : (j // 2) * 2 + 2, :] = 1
        else:
            grid[:, (j // 2) * 2 + 1 : (j // 2) * 2 + 3] = 1
        masks[j] = grid.ravel()
    z = rng.binomial(1, 0.3, size=(n, k)).astype(float)
    on = np.clip(z @ masks, 0, 1)
    p = 0.9 * on + 0.05 * (1 - on)
    X = rng.binomial(1, p).astype(np.float64)
    return X, z, masks


# --------------------------------------------------------------------------------------
# Molecular biology: Ramachandran-style torsion angles
# --------------------------------------------------------------------------------------
def torsion_angles(seed: int = 0, n: int = 4000) -> np.ndarray:
    """(phi, psi) backbone torsion angles in radians, scaled to roughly [-pi, pi].

    Three modes: alpha-helix (-1.1, -0.7), beta-sheet (-2.1, 2.3), left-handed helix
    (1.0, 0.8). Returns array [n, 2]."""
    rng = _rng(seed)
    centres = np.array([[-1.1, -0.7], [-2.1, 2.3], [1.0, 0.8]])
    scales = np.array([[0.25, 0.3], [0.35, 0.35], [0.2, 0.2]])
    w = np.array([0.55, 0.35, 0.10])
    z = rng.choice(3, size=n, p=w)
    x = centres[z] + scales[z] * rng.normal(size=(n, 2))
    return x


# --------------------------------------------------------------------------------------
# Marketing: ad-spend structural causal model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AdSpendSCM:
    """Linear-Gaussian SCM.

    S = seasonality ~ N(0, 1)
    A = ad spend     = a_s S + U_A,   U_A ~ N(0, sd_a^2)
    T = site traffic = b_s S + b_a A + U_T
    C = conversions  = c_t T + c_s S + U_C
    """

    a_s: float = 0.8
    b_s: float = 0.5
    b_a: float = 1.5
    c_t: float = 0.6
    c_s: float = 0.3
    sd_a: float = 1.0
    sd_t: float = 0.7
    sd_c: float = 0.5


def ad_spend_observational(seed: int = 0, n: int = 2000, scm: AdSpendSCM = AdSpendSCM()) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    s = rng.normal(size=n)
    u_a = scm.sd_a * rng.normal(size=n)
    u_t = scm.sd_t * rng.normal(size=n)
    u_c = scm.sd_c * rng.normal(size=n)
    a = scm.a_s * s + u_a
    t = scm.b_s * s + scm.b_a * a + u_t
    c = scm.c_t * t + scm.c_s * s + u_c
    return {"S": s, "A": a, "T": t, "C": c, "U_A": u_a, "U_T": u_t, "U_C": u_c}
