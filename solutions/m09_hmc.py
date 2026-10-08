"""Module 9 — Metropolis–Hastings and Hamiltonian dynamics.

A functional HMC sampler. Targets are plain functions `log_prob(z) -> scalar` of a flat
vector z in R^d; gradients come from jax.grad. Mass matrices are diagonal and passed as
`inv_mass`, a vector of length d (None means the identity).

Targets used throughout Day 3 are defined at the bottom: a 1-D standard normal, a
correlated 2-D Gaussian, and the hierarchical regional-uplift model (eight-schools
structure) in centred and non-centred parameterisations.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .data import regional_uplift

Array = jax.Array
LogProb = Callable[[Array], Array]


class HMCInfo(NamedTuple):
    """Per-iteration diagnostics of one HMC transition."""

    accept: Array  # bool: was the proposal accepted
    accept_prob: Array  # min(1, exp(-energy_error))
    energy_error: Array  # H(q*, p*) - H(q, p); large positive values flag divergences
    log_prob: Array  # log target density at the returned state


# --------------------------------------------------------------------------------------
# Step 1: random-walk Metropolis–Hastings
# --------------------------------------------------------------------------------------
def random_walk_mh(log_prob: LogProb, key: Array, x0: Array, n_steps: int, step_size: float) -> tuple[Array, Array]:
    """Random-walk Metropolis with isotropic Gaussian proposals of scale `step_size`.

    Returns (samples [n_steps, d], acceptance rate). Implement with lax.scan; draw one
    proposal key and one uniform key per step from a split of the carried key. Accept
    with probability min(1, p(x') / p(x)) computed in log space.
    """
    # [m09 step 1]

    def body(carry, _):
        x, lp, key = carry
        key, k_prop, k_u = jax.random.split(key, 3)
        x_prop = x + step_size * jax.random.normal(k_prop, x.shape)
        lp_prop = log_prob(x_prop)
        log_u = jnp.log(jax.random.uniform(k_u))
        accept = log_u < lp_prop - lp
        x_new = jnp.where(accept, x_prop, x)
        lp_new = jnp.where(accept, lp_prop, lp)
        return (x_new, lp_new, key), (x_new, accept)

    (_, _, _), (xs, accepts) = lax.scan(body, (x0, log_prob(x0), key), None, length=n_steps)
    return xs, jnp.mean(accepts)


# --------------------------------------------------------------------------------------
# Step 2: the leapfrog integrator
# --------------------------------------------------------------------------------------
def leapfrog(grad_log_prob: Callable[[Array], Array], q: Array, p: Array, step_size: float, n_steps, inv_mass: Array) -> tuple[Array, Array]:
    """Integrate Hamilton's equations for H = -log p(q) + p^T M^{-1} p / 2 with the
    Störmer–Verlet (leapfrog) scheme for `n_steps` steps of size `step_size`.

    One step: p <- p + (eps/2) grad log p(q);  q <- q + eps M^{-1} p;
              p <- p + (eps/2) grad log p(q).
    `n_steps` may be a traced integer, so use lax.fori_loop. Returns (q, p).
    """
    # [m09 step 2]

    def step(_, qp):
        q, p = qp
        p = p + 0.5 * step_size * grad_log_prob(q)
        q = q + step_size * inv_mass * p
        p = p + 0.5 * step_size * grad_log_prob(q)
        return (q, p)

    return lax.fori_loop(0, n_steps, step, (q, p))


# --------------------------------------------------------------------------------------
# Step 3: the Hamiltonian and one HMC transition
# --------------------------------------------------------------------------------------
def kinetic_energy(p: Array, inv_mass: Array) -> Array:
    """K(p) = p^T M^{-1} p / 2 for a diagonal mass matrix."""
    # [m09 step 3]
    return 0.5 * jnp.sum(inv_mass * p**2)


def hamiltonian(q: Array, p: Array, log_prob: LogProb, inv_mass: Array) -> Array:
    """H(q, p) = -log p(q) + K(p)."""
    # [m09 step 3]
    return -log_prob(q) + kinetic_energy(p, inv_mass)


def hmc_step(log_prob: LogProb, key: Array, q: Array, step_size: float, n_leapfrog, inv_mass: Array) -> tuple[Array, HMCInfo]:
    """One HMC transition.

    Draw p ~ N(0, M) (so p = sqrt(M) eps = eps / sqrt(inv_mass)), integrate with
    leapfrog, negate the momentum (not needed for the acceptance test but part of the
    reversible map), accept with probability min(1, exp(-(H* - H))). Non-finite
    energy errors must be treated as +inf (reject). Returns (q_new, HMCInfo).
    """
    # [m09 step 3]
    k_p, k_u = jax.random.split(key)
    p = jax.random.normal(k_p, q.shape) / jnp.sqrt(inv_mass)
    grad = jax.grad(log_prob)
    q_new, p_new = leapfrog(grad, q, p, step_size, n_leapfrog, inv_mass)
    p_new = -p_new
    h0 = hamiltonian(q, p, log_prob, inv_mass)
    h1 = hamiltonian(q_new, p_new, log_prob, inv_mass)
    energy_error = h1 - h0
    energy_error = jnp.where(jnp.isfinite(energy_error), energy_error, jnp.inf)
    accept_prob = jnp.minimum(1.0, jnp.exp(-energy_error))
    accept = jax.random.uniform(k_u) < accept_prob
    q_out = jnp.where(accept, q_new, q)
    lp_out = jnp.where(accept, -h1 + kinetic_energy(p_new, inv_mass), -h0 + kinetic_energy(p, inv_mass))
    return q_out, HMCInfo(accept=accept, accept_prob=accept_prob, energy_error=energy_error, log_prob=lp_out)


# --------------------------------------------------------------------------------------
# Step 4: a fixed-settings sampler
# --------------------------------------------------------------------------------------
def hmc(log_prob: LogProb, key: Array, q0: Array, n_samples: int, step_size: float, n_leapfrog: int, inv_mass: Array | None = None) -> tuple[Array, HMCInfo]:
    """Run `n_samples` HMC transitions from q0 with fixed step size, trajectory length
    and (diagonal) inverse mass. Returns (samples [n_samples, d], HMCInfo with a
    leading axis of length n_samples). Use lax.scan."""
    # [m09 step 4]
    if inv_mass is None:
        inv_mass = jnp.ones_like(q0)

    def body(carry, _):
        q, key = carry
        key, sub = jax.random.split(key)
        q, info = hmc_step(log_prob, sub, q, step_size, n_leapfrog, inv_mass)
        return (q, key), (q, info)

    (_, _), (qs, infos) = lax.scan(body, (q0, key), None, length=n_samples)
    return qs, infos


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def standard_normal_log_prob(z: Array) -> Array:
    return -0.5 * jnp.sum(z**2)


GAUSS2D_COV = jnp.array([[1.0, 0.9], [0.9, 2.0]])
GAUSS2D_PREC = jnp.linalg.inv(GAUSS2D_COV)


def gaussian_2d_log_prob(z: Array) -> Array:
    """Correlated 2-D Gaussian with covariance GAUSS2D_COV, zero mean."""
    return -0.5 * z @ GAUSS2D_PREC @ z


_UPLIFT_Y, _UPLIFT_SIGMA = (jnp.asarray(a) for a in regional_uplift())
UPLIFT_DIM = 10  # [mu, log_tau, theta_1..8] or [mu, log_tau, eta_1..8]


def _log_half_normal(x: Array, scale: float) -> Array:
    return jnp.log(2.0) - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(scale) - 0.5 * (x / scale) ** 2


def uplift_centered_log_prob(z: Array) -> Array:
    """Hierarchical regional-uplift model, centred parameterisation.

    z = [mu, log tau, theta_1, ..., theta_8].
    mu ~ N(0, 5^2), tau ~ HalfNormal(5), theta_j ~ N(mu, tau^2), y_j ~ N(theta_j, sigma_j^2).
    Includes the log-Jacobian of tau = exp(log tau).
    """
    mu, log_tau, theta = z[0], z[1], z[2:]
    tau = jnp.exp(log_tau)
    lp = -0.5 * (mu / 5.0) ** 2
    lp = lp + _log_half_normal(tau, 5.0) + log_tau
    lp = lp + jnp.sum(-0.5 * ((theta - mu) / tau) ** 2 - log_tau)
    lp = lp + jnp.sum(-0.5 * ((_UPLIFT_Y - theta) / _UPLIFT_SIGMA) ** 2)
    return lp


def uplift_noncentered_log_prob(z: Array) -> Array:
    """Same model, non-centred: z = [mu, log tau, eta_1..8] with theta_j = mu + tau eta_j,
    eta_j ~ N(0, 1)."""
    mu, log_tau, eta = z[0], z[1], z[2:]
    tau = jnp.exp(log_tau)
    theta = mu + tau * eta
    lp = -0.5 * (mu / 5.0) ** 2
    lp = lp + _log_half_normal(tau, 5.0) + log_tau
    lp = lp + jnp.sum(-0.5 * eta**2)
    lp = lp + jnp.sum(-0.5 * ((_UPLIFT_Y - theta) / _UPLIFT_SIGMA) ** 2)
    return lp


def uplift_noncentered_to_centered(z: Array) -> Array:
    """Map non-centred samples [..., 10] to [mu, log tau, theta_1..8]."""
    mu, log_tau, eta = z[..., :1], z[..., 1:2], z[..., 2:]
    return jnp.concatenate([mu, log_tau, mu + jnp.exp(log_tau) * eta], axis=-1)
