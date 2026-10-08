"""Module 15b — Score-based diffusion as a continuous-time latent-variable model.

A variance-preserving SDE  dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW  carries data x_0 to
(nearly) N(0, I) at t = 1. Its time reversal (Anderson 1982) is a generative model whose
only unknown is the score  grad_x log p_t(x). Denoising score matching learns the score
from the closed-form perturbation kernel; Euler-Maruyama on the reverse SDE draws
samples; the probability-flow ODE gives exact log-likelihoods through the instantaneous
change-of-variables formula. The forward SDE plays the role of a fixed variational
posterior over the latent path, and the weighted score-matching objective is an ELBO.

Conventions: x is a single d-vector and t a scalar in [T_EPS, 1]; use jax.vmap for
batches. Data: Ramachandran-style (phi, psi) torsion angles from data.torsion_angles().
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from .m08_bbvi import adam_init, adam_update
from .m14_vae import init_mlp, mlp

Array = jax.Array
ScoreFn = Callable[[Array, Array], Array]  # (x [d], t scalar) -> score [d]
NetParams = list[dict[str, Array]]  # Module 14's MLP layers

BETA_MIN = 0.1
BETA_MAX = 20.0
T_EPS = 1e-3


# --------------------------------------------------------------------------------------
# Step 1: the VP-SDE and its perturbation kernel
# --------------------------------------------------------------------------------------
def beta(t: Array) -> Array:
    """Linear noise schedule beta(t) = BETA_MIN + t (BETA_MAX - BETA_MIN), t in [0, 1]."""
    raise NotImplementedError  # Module 15b, Step 1


def alpha_bar(t: Array) -> Array:
    """alpha_bar(t) = exp(-int_0^t beta(s) ds), the squared signal coefficient of the
    perturbation kernel. Integrate beta in closed form."""
    raise NotImplementedError  # Module 15b, Step 1


def perturbation_kernel(x0: Array, t: Array) -> tuple[Array, Array]:
    """Mean and (scalar) standard deviation of p_t(x_t | x_0) = N(sqrt(ab) x0, (1-ab) I)."""
    raise NotImplementedError  # Module 15b, Step 1


def sample_forward(key: Array, x0: Array, t: Array) -> Array:
    """One draw of x_t given x_0 via the reparameterisation mean + std * eps."""
    raise NotImplementedError  # Module 15b, Step 1


# --------------------------------------------------------------------------------------
# Step 2: exact scores for Gaussian-mixture data
# --------------------------------------------------------------------------------------
def perturbed_mixture_log_density(x: Array, t: Array, weights: Array, means: Array, covs: Array) -> Array:
    """log p_t(x) when p_0 is a Gaussian mixture: each component N(m_k, S_k) becomes
    N(sqrt(ab) m_k, ab S_k + (1 - ab) I). Provided as the reference for Step 2."""
    ab = alpha_bar(t)
    d = x.shape[0]

    def comp(w, m, S):
        mt = jnp.sqrt(ab) * m
        St = ab * S + (1.0 - ab) * jnp.eye(d)
        return jnp.log(w) + jax.scipy.stats.multivariate_normal.logpdf(x, mt, St)

    return jax.nn.logsumexp(jax.vmap(comp)(weights, means, covs))


def gaussian_mixture_score(x: Array, t: Array, weights: Array, means: Array, covs: Array) -> Array:
    """grad_x log p_t(x) for Gaussian-mixture data, in closed form: the responsibility-
    weighted sum of the perturbed components' Gaussian scores -S_k(t)^{-1}(x - m_k(t)).
    Do not call jax.grad here; the test compares you against it."""
    raise NotImplementedError  # Module 15b, Step 2


# --------------------------------------------------------------------------------------
# Step 3: score network and denoising score matching
# --------------------------------------------------------------------------------------
def time_features(t: Array, n_freqs: int = 8) -> Array:
    """[t, sin(2 pi 2^k t), cos(2 pi 2^k t)] for k = 0..n_freqs-1: shape [1 + 2 n_freqs]."""
    freqs = 2.0 ** jnp.arange(n_freqs)
    ang = 2.0 * jnp.pi * freqs * t
    return jnp.concatenate([jnp.reshape(t, (1,)), jnp.sin(ang), jnp.cos(ang)])


def init_score_net(key: Array, dim: int = 2, hidden: int = 128, n_freqs: int = 8) -> NetParams:
    """Module 14's init_mlp with sizes [dim + 1 + 2 n_freqs, hidden, hidden, dim]: the
    input is [x, time_features(t)]."""
    raise NotImplementedError  # Module 15b, Step 3


def score_net(params: NetParams, x: Array, t: Array) -> Array:
    """s_theta(x, t) = mlp(params, [x, time_features(t)], activation=jax.nn.silu) / std(t),
    with Module 14's mlp. Recover n_freqs from the first layer's input width. Dividing
    by the kernel std gives the output the right scale near t = 0 where the true score
    is O(1/std)."""
    raise NotImplementedError  # Module 15b, Step 3


def make_score_fn(params: NetParams) -> ScoreFn:
    return lambda x, t: score_net(params, x, t)


def dsm_loss_fn(score_fn: ScoreFn, key: Array, x0: Array) -> Array:
    """Denoising score-matching loss with weighting lambda(t) = std(t)^2 for a batch
    x0 [n, d]: draw t ~ U(T_EPS, 1) and eps ~ N(0, I) per row, form x_t, and average
        || std(t) s(x_t, t) + eps ||^2
    over the batch (the target score of the kernel is -eps / std(t))."""
    raise NotImplementedError  # Module 15b, Step 3


def dsm_loss(params: NetParams, key: Array, x0: Array) -> Array:
    """dsm_loss_fn with the score network."""
    raise NotImplementedError  # Module 15b, Step 3


# --------------------------------------------------------------------------------------
# Step 4: training and the reverse-time sampler
# --------------------------------------------------------------------------------------
def train_score_model(key: Array, X: Array, n_steps: int, batch_size: int = 256, lr: float = 1e-3, hidden: int = 128) -> tuple[NetParams, Array]:
    """Minimise dsm_loss with Adam (descent: pass -grad to m08's ascent update) over
    minibatches sampled with replacement, in one lax.scan. Returns (params, loss_trace)."""
    raise NotImplementedError  # Module 15b, Step 4


def reverse_drift(score_fn: ScoreFn, x: Array, t: Array) -> Array:
    """Drift of the reverse-time SDE run backwards in t:  f(x,t) - g(t)^2 s(x,t)
    with f = -1/2 beta x and g^2 = beta."""
    raise NotImplementedError  # Module 15b, Step 4


def sample_reverse(key: Array, score_fn: ScoreFn, n: int, n_steps: int, dim: int = 2) -> Array:
    """Euler-Maruyama on the reverse SDE from t = 1 to t = T_EPS with n_steps uniform
    steps, starting from x_1 ~ N(0, I). Returns [n, dim]. The final step is the
    denoised mean (no noise added)."""
    raise NotImplementedError  # Module 15b, Step 4


# --------------------------------------------------------------------------------------
# Step 5: probability-flow ODE and exact likelihood
# --------------------------------------------------------------------------------------
def probability_flow_drift(score_fn: ScoreFn, x: Array, t: Array) -> Array:
    """Drift of the deterministic ODE with the same marginals as the SDE:
    f(x,t) - 1/2 g(t)^2 s(x,t) = -1/2 beta(t) (x + s(x,t))."""
    raise NotImplementedError  # Module 15b, Step 5


def _rk4(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _ode_log_likelihood(score_fn: ScoreFn, x: Array, n_steps: int, divergence: Callable[[Array, Array], Array]) -> Array:
    """Integrate (x, log-det) from T_EPS to 1 with RK4 and add log N(x_1; 0, I)."""
    d = x.shape[0]
    dt = (1.0 - T_EPS) / n_steps
    ts = T_EPS + dt * jnp.arange(n_steps)

    def rhs(y, t):
        xx = y[:d]
        return jnp.concatenate([probability_flow_drift(score_fn, xx, t), jnp.reshape(divergence(xx, t), (1,))])

    def step(y, t):
        return _rk4(rhs, y, t, dt), None

    y0 = jnp.concatenate([x, jnp.zeros(1)])
    y1, _ = lax.scan(step, y0, ts)
    x1, int_div = y1[:d], y1[d]
    log_prior = jax.scipy.stats.multivariate_normal.logpdf(x1, jnp.zeros(d), jnp.eye(d))
    return log_prior + int_div


def ode_log_likelihood_exact(score_fn: ScoreFn, x: Array, n_steps: int = 200) -> Array:
    """log p_0(x) = log p_1(x_1) + int_0^1 div f_ode(x_t, t) dt, with the divergence
    computed exactly as the trace of jax.jacfwd of the ODE drift (fine in 2-D)."""
    raise NotImplementedError  # Module 15b, Step 5


def ode_log_likelihood(score_fn: ScoreFn, x: Array, key: Array, n_steps: int = 200) -> Array:
    """Same as ode_log_likelihood_exact but with Hutchinson's estimator of the divergence,
    eps^T J eps for a single Rademacher eps drawn once per data point and reused along
    the trajectory, evaluated with jax.jvp (no Jacobian is formed)."""
    raise NotImplementedError  # Module 15b, Step 5
