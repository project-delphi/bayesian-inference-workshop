"""Module 7 — Gradient estimators for the ELBO.

Target: the posterior of a Bayesian logistic regression for SaaS churn,
    beta ~ N(0, prior_scale^2 I),   y_i ~ Bernoulli(sigmoid(x_i . beta)).
Guide: mean-field Gaussian q_phi(z) = N(loc, diag(exp(log_scale))^2) with
phi = {"loc": [d], "log_scale": [d]}.

Two unbiased estimators of grad_phi ELBO(phi) = grad_phi E_q[ log p(x, z) - log q_phi(z) ]:

  score function (REINFORCE):  E_q[ (log p(x,z) - log q(z)) grad_phi log q_phi(z) ]
  pathwise (reparameterisation): E_eps[ grad_phi ( log p(x, g_phi(eps)) - log q_phi(g_phi(eps)) ) ],
                                 z = g_phi(eps) = loc + exp(log_scale) * eps.

The Laplace approximation serves as the reference posterior.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict[str, Array]


# --------------------------------------------------------------------------------------
# Step 1: model and Laplace reference
# --------------------------------------------------------------------------------------
def churn_log_joint(beta: Array, X: Array, y: Array, prior_scale: float = 2.5) -> Array:
    """log p(y | X, beta) + log p(beta) for Bernoulli-logit likelihood and N(0, s^2 I) prior.

    Use jax.nn.log_sigmoid for the likelihood: log p(y_i) = y_i log s(eta_i) + (1 - y_i) log s(-eta_i)."""
    # [m07 step 1]
    eta = X @ beta
    ll = jnp.sum(y * jax.nn.log_sigmoid(eta) + (1 - y) * jax.nn.log_sigmoid(-eta))
    d = beta.shape[0]
    lp = -0.5 * jnp.sum(beta**2) / prior_scale**2 - 0.5 * d * jnp.log(2 * jnp.pi * prior_scale**2)
    return ll + lp


def laplace(log_joint: Callable[[Array], Array], z0: Array, n_newton: int = 25) -> tuple[Array, Array]:
    """Laplace approximation: Newton's method to the mode of log_joint from z0, then
    cov = (-Hessian)^{-1} at the mode. Returns (mean, cov). Use lax.fori_loop."""
    # [m07 step 1]
    g = jax.grad(log_joint)
    H = jax.hessian(log_joint)

    def step(_, z):
        return z - jnp.linalg.solve(H(z), g(z))

    mode = jax.lax.fori_loop(0, n_newton, step, z0)
    cov = jnp.linalg.inv(-H(mode))
    return mode, cov


def laplace_log_evidence(log_joint: Callable[[Array], Array], mean: Array, cov: Array) -> Array:
    """log p(x) ~= log_joint(mean) + d/2 log(2 pi) + 1/2 log det cov."""
    # [m07 step 1]
    d = mean.shape[0]
    return log_joint(mean) + 0.5 * d * jnp.log(2 * jnp.pi) + 0.5 * jnp.linalg.slogdet(cov)[1]


# --------------------------------------------------------------------------------------
# Mean-field Gaussian guide (given)
# --------------------------------------------------------------------------------------
def guide_sample(key: Array, params: Params, n: int) -> Array:
    eps = jax.random.normal(key, (n, params["loc"].shape[0]))
    return params["loc"] + jnp.exp(params["log_scale"]) * eps


def guide_log_prob(params: Params, z: Array) -> Array:
    scale = jnp.exp(params["log_scale"])
    return jnp.sum(-0.5 * ((z - params["loc"]) / scale) ** 2 - params["log_scale"] - 0.5 * jnp.log(2 * jnp.pi))


# --------------------------------------------------------------------------------------
# Step 2: score-function estimator
# --------------------------------------------------------------------------------------
def elbo_grad_score(key: Array, params: Params, log_joint: Callable[[Array], Array], n: int) -> Params:
    """REINFORCE estimate of grad_phi ELBO with n samples and no baseline:
        mean_s [ f(z_s) * grad_phi log q_phi(z_s) ],  f(z) = log_joint(z) - log q_phi(z).
    z_s must be treated as constants (use jax.lax.stop_gradient on the samples)."""
    # [m07 step 2]
    z = jax.lax.stop_gradient(guide_sample(key, params, n))
    f = jax.vmap(log_joint)(z) - jax.vmap(guide_log_prob, in_axes=(None, 0))(params, z)
    score = jax.vmap(jax.grad(guide_log_prob), in_axes=(None, 0))(params, z)  # pytree of [n, d]
    return jax.tree_util.tree_map(lambda s: jnp.mean(f[:, None] * s, axis=0), score)


# --------------------------------------------------------------------------------------
# Step 3: score-function with a control variate
# --------------------------------------------------------------------------------------
def elbo_grad_score_cv(key: Array, params: Params, log_joint: Callable[[Array], Array], n: int) -> Params:
    """Score-function estimator with the per-coordinate optimal scalar baseline
        a_j = Cov(f s_j, s_j) / Var(s_j),  estimate = mean_s[(f(z_s) - a_j) s_j(z_s)],
    where s_j is coordinate j of grad_phi log q_phi(z). The baseline is estimated from the
    same samples (Ranganath et al. 2014, Section 3.2)."""
    # [m07 step 3]
    z = jax.lax.stop_gradient(guide_sample(key, params, n))
    f = jax.vmap(log_joint)(z) - jax.vmap(guide_log_prob, in_axes=(None, 0))(params, z)
    score = jax.vmap(jax.grad(guide_log_prob), in_axes=(None, 0))(params, z)

    def cv(s):  # s: [n, d]
        fs = f[:, None] * s
        cov = jnp.mean((fs - fs.mean(0)) * (s - s.mean(0)), axis=0)
        var = jnp.var(s, axis=0)
        a = cov / var
        return jnp.mean((f[:, None] - a[None, :]) * s, axis=0)

    return jax.tree_util.tree_map(cv, score)


# --------------------------------------------------------------------------------------
# Step 4: pathwise (reparameterisation) estimator
# --------------------------------------------------------------------------------------
def elbo_grad_reparam(key: Array, params: Params, log_joint: Callable[[Array], Array], n: int) -> Params:
    """grad_phi of mean_s[ log_joint(g_phi(eps_s)) - log q_phi(g_phi(eps_s)) ] with eps fixed."""
    # [m07 step 4]
    eps = jax.random.normal(key, (n, params["loc"].shape[0]))

    def objective(p):
        z = p["loc"] + jnp.exp(p["log_scale"]) * eps
        return jnp.mean(jax.vmap(log_joint)(z) - jax.vmap(guide_log_prob, in_axes=(None, 0))(p, z))

    return jax.grad(objective)(params)


# --------------------------------------------------------------------------------------
# Step 5: variance study
# --------------------------------------------------------------------------------------
def estimator_variance(key: Array, estimator, params: Params, log_joint, n_samples: int, n_repeats: int) -> Array:
    """Run `estimator` n_repeats times with independent keys and return the per-coordinate
    variance of the flattened gradient, shape [2 d]."""
    # [m07 step 5]
    keys = jax.random.split(key, n_repeats)

    def one(k):
        g = estimator(k, params, log_joint, n_samples)
        return jnp.concatenate([g["loc"], g["log_scale"]])

    G = jax.vmap(one)(keys)
    return jnp.var(G, axis=0)


def gradient_variance_study(key: Array, params: Params, log_joint, n_samples_list: tuple[int, ...], n_repeats: int = 200) -> dict[str, Array]:
    """Total gradient variance (sum over coordinates) for each estimator and each n in
    n_samples_list. Returns {"score": [...], "score_cv": [...], "reparam": [...]}."""
    # [m07 step 5]
    out = {}
    for name, est in [("score", elbo_grad_score), ("score_cv", elbo_grad_score_cv), ("reparam", elbo_grad_reparam)]:
        vals = []
        for i, n in enumerate(n_samples_list):
            vals.append(estimator_variance(jax.random.fold_in(key, i), est, params, log_joint, n, n_repeats).sum())
        out[name] = jnp.stack(vals)
    return out
