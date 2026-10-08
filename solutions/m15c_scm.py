"""Module 15c — Structural causal models and counterfactual inference on the PPL.

A structural causal model (SCM) is a probabilistic program of a particular shape: every
endogenous variable is a deterministic function of its parents and one exogenous noise
variable, and only the noise variables are random. Written with the Module 12
primitives, the noise variables are `sample` sites and the endogenous variables are
`deterministic` sites. Two things then become handler operations:

- an intervention do(X := x) is a handler that replaces the value of site X and lets the
  downstream mechanisms run with it;
- a counterfactual is abduction (posterior inference over the noise given what was
  observed, using the Module 13 engines), action (the `do` handler) and prediction
  (running the program forward with the abducted noise).

The running example is the linear-Gaussian ad-spend model `data.AdSpendSCM`:

    S  ~ N(0, 1)                      seasonality (confounder)
    A  = a_s S + U_A                  ad spend
    T  = b_s S + b_a A + U_T          site traffic
    C  = c_t T + c_s S + U_C          conversions

Linear-Gaussian structure means every quantity has a closed form against which the
handler machinery is tested.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from .data import AdSpendSCM
from .m02_diagnostic import gaussian_condition
from .m12_handlers import Messenger, Normal, _new_message, apply_stack, _HANDLER_STACK, sample, seed, substitute, trace
from .m13_ppl_inference import hmc, posterior_samples_svi, svi

Array = jax.Array

NOISE_SITES = ("S", "U_A", "U_T", "U_C")
ENDOGENOUS = ("S", "A", "T", "C")


# ======================================================================================
# Step 1: a `deterministic` primitive and the SCM as a program
# ======================================================================================
def deterministic(name: str, value: Array) -> Array:
    """Primitive: record a named deterministic node.

    Builds a message with type "deterministic", fn None, the given value and
    is_observed True, sends it down the handler stack and returns msg["value"]. The
    Module 12 handlers leave it alone: `seed`, `condition` and `replay` act only on
    "sample" messages, `log_density` sums only "sample" sites, and `trace` records it.
    `substitute` and `block` act by name, which is what `do` below relies on. With no
    handlers on the stack the value is returned unchanged.
    """
    # [m15c step 1]
    if not _HANDLER_STACK:
        return value
    msg = _new_message("deterministic", name, value=value, is_observed=True)
    return apply_stack(msg)["value"]


def ad_spend_scm(scm: AdSpendSCM) -> dict[str, Array]:
    """The ad-spend SCM as a program. Noise sites S, U_A, U_T, U_C are `sample` sites;
    A, T, C are `deterministic` sites computed from the values the primitives RETURN (so
    an intervention on A propagates to T and C). Returns a dict of the four endogenous
    values."""
    # [m15c step 1]
    S = sample("S", Normal(0.0, 1.0))
    U_A = sample("U_A", Normal(0.0, scm.sd_a))
    U_T = sample("U_T", Normal(0.0, scm.sd_t))
    U_C = sample("U_C", Normal(0.0, scm.sd_c))
    A = deterministic("A", scm.a_s * S + U_A)
    T = deterministic("T", scm.b_s * S + scm.b_a * A + U_T)
    C = deterministic("C", scm.c_t * T + scm.c_s * S + U_C)
    return {"S": S, "A": A, "T": T, "C": C}


def run_scm(model: Callable, key: Array, n: int, *args) -> dict[str, Array]:
    """Run `model` n times under trace+seed with independent keys (jax.vmap over keys)
    and return every recorded site's value with a leading axis of length n."""
    # [m15c step 1]

    def one(k):
        tr = trace(seed(model, k)).get_trace(*args)
        return {name: site["value"] for name, site in tr.items()}

    return jax.vmap(one)(jax.random.split(key, n))


def structural_matrix(scm: AdSpendSCM) -> tuple[Array, Array]:
    """The linear map x = M u from noise u = (S, U_A, U_T, U_C) to endogenous
    x = (S, A, T, C), and the noise standard deviations (1, sd_a, sd_t, sd_c). Solve the
    structural equations by substitution so each row of M is in terms of u only."""
    # [m15c step 1]
    a_s, b_s, b_a, c_t, c_s = scm.a_s, scm.b_s, scm.b_a, scm.c_t, scm.c_s
    row_S = jnp.array([1.0, 0.0, 0.0, 0.0])
    row_A = jnp.array([a_s, 1.0, 0.0, 0.0])
    row_T = b_s * row_S + b_a * row_A + jnp.array([0.0, 0.0, 1.0, 0.0])
    row_C = c_t * row_T + c_s * row_S + jnp.array([0.0, 0.0, 0.0, 1.0])
    M = jnp.stack([row_S, row_A, row_T, row_C])
    noise_sd = jnp.array([1.0, scm.sd_a, scm.sd_t, scm.sd_c])
    return M, noise_sd


# ======================================================================================
# Step 2: the `do` handler and interventional versus observational quantities
# ======================================================================================
class do(Messenger):
    """Intervene: do(fn, {"A": 2.0}) replaces the value at site A.

    For a "deterministic" site the value is overwritten and msg["intervened"] = True; the
    model code uses the returned value, so every descendant is computed from the
    intervened value. For a "sample" site (a noise variable) the site is converted into a
    deterministic one: type "deterministic", fn None, is_observed True, value set. It
    therefore contributes nothing to log_density and is never resampled by seed. Put
    `do` innermost (directly around the model) so other handlers see the converted
    message.
    """

    def __init__(self, fn: Callable | None = None, interventions: dict[str, Array] | None = None):
        super().__init__(fn)
        self.interventions = interventions or {}

    def process_message(self, msg):
        # [m15c step 2]
        if msg["name"] not in self.interventions:
            return
        value = jnp.asarray(self.interventions[msg["name"]])
        if msg["type"] == "sample":
            msg["type"] = "deterministic"
            msg["fn"] = None
            msg["key"] = None
        msg["value"] = value
        msg["is_observed"] = True
        msg["intervened"] = True


def interventional_samples(scm: AdSpendSCM, interventions: dict[str, Array], key: Array, n: int) -> dict[str, Array]:
    """Samples of every site from p(. | do(interventions)): run_scm of do(ad_spend_scm)."""
    # [m15c step 2]
    return run_scm(do(ad_spend_scm, interventions), key, n, scm)


def observational_slope(a: Array, c: Array) -> Array:
    """Least-squares slope of C on A: cov(A, C) / var(A). This is E[C | A = a]'s slope."""
    # [m15c step 2]
    a_c = a - a.mean()
    return jnp.sum(a_c * (c - c.mean())) / jnp.sum(a_c**2)


def backdoor_slope(a: Array, c: Array, s: Array) -> Array:
    """Coefficient of A in the least-squares regression of C on (1, A, S). Adjusting for
    the back-door variable S recovers the causal effect dE[C | do(A = a)] / da."""
    # [m15c step 2]
    X = jnp.stack([jnp.ones_like(a), a, s], axis=1)
    beta, *_ = jnp.linalg.lstsq(X, c, rcond=None)
    return beta[1]


def causal_effect(scm: AdSpendSCM) -> Array:
    """dE[C | do(A = a)] / da = c_t b_a, read off the mechanisms."""
    # [m15c step 2]
    return jnp.asarray(scm.c_t * scm.b_a)


def confounding_bias(scm: AdSpendSCM) -> Array:
    """observational_slope - causal_effect in closed form:
    (c_t b_s + c_s) * a_s / (a_s^2 + sd_a^2), the path A <- S -> {T, C} rescaled by var(A)."""
    # [m15c step 2]
    return jnp.asarray((scm.c_t * scm.b_s + scm.c_s) * scm.a_s / (scm.a_s**2 + scm.sd_a**2))


# ======================================================================================
# Step 3: abduction-action-prediction with full observation
# ======================================================================================
def abduct_exact(scm: AdSpendSCM, obs: dict[str, Array]) -> dict[str, Array]:
    """Invert the structural equations: given S, A, T, C return the noise values
    S, U_A, U_T, U_C that produced them. Exact because each mechanism is additive in its
    noise term."""
    # [m15c step 3]
    S, A, T, C = (jnp.asarray(obs[k]) for k in ENDOGENOUS)
    return {
        "S": S,
        "U_A": A - scm.a_s * S,
        "U_T": T - scm.b_s * S - scm.b_a * A,
        "U_C": C - scm.c_t * T - scm.c_s * S,
    }


def run_with(model: Callable, values: dict[str, Array], *args) -> dict[str, Array]:
    """Run `model` once with the given site values substituted (no key needed when every
    sample site is covered) and return every recorded site's value."""
    # [m15c step 3]
    tr = trace(substitute(model, values)).get_trace(*args)
    return {name: site["value"] for name, site in tr.items()}


def counterfactual(scm: AdSpendSCM, obs: dict[str, Array], interventions: dict[str, Array]) -> dict[str, Array]:
    """Pearl's three steps with a fully observed world: abduct the noise exactly, apply
    `do`, predict by running the program. Returns every site of the counterfactual
    world."""
    # [m15c step 3]
    noise = abduct_exact(scm, obs)
    return run_with(do(ad_spend_scm, interventions), noise, scm)


def counterfactual_closed_form(scm: AdSpendSCM, obs: dict[str, Array], a_new: Array) -> Array:
    """C under do(A := a_new) for a fully observed week:
    c_t (b_s S + b_a a_new + U_T) + c_s S + U_C with U_T, U_C from abduct_exact."""
    # [m15c step 3]
    u = abduct_exact(scm, obs)
    S = jnp.asarray(obs["S"])
    T_cf = scm.b_s * S + scm.b_a * jnp.asarray(a_new) + u["U_T"]
    return scm.c_t * T_cf + scm.c_s * S + u["U_C"]


# ======================================================================================
# Step 4: abduction as posterior inference (partial observation)
# ======================================================================================
def abduction_model(scm: AdSpendSCM, obs: dict[str, Array]) -> None:
    """Posterior over the noise given spend A and traffic T for one week, with
    seasonality S and the conversion noise U_C latent.

    The observations enter as *implied noise*: since A = a_s S + U_A is additive in U_A,
    observing A is the same as observing U_A = A - a_s S, so write
    sample("U_A", Normal(0, sd_a), obs=A - a_s S). The Jacobian of this change of
    variables is 1. U_C is latent with its prior. The latent sites are S and U_C."""
    # [m15c step 4]
    A, T = jnp.asarray(obs["A"]), jnp.asarray(obs["T"])
    S = sample("S", Normal(0.0, 1.0))
    sample("U_A", Normal(0.0, scm.sd_a), obs=A - scm.a_s * S)
    sample("U_T", Normal(0.0, scm.sd_t), obs=T - scm.b_s * S - scm.b_a * A)
    sample("U_C", Normal(0.0, scm.sd_c))


def exact_counterfactual_C(scm: AdSpendSCM, obs: dict[str, Array], a_new: Array) -> tuple[Array, Array]:
    """Mean and standard deviation of C under do(A := a_new) given A and T observed.

    Condition the joint Gaussian of (S, A, T) on (A, T) with Module 2's
    gaussian_condition; then C_cf = c_t (T + b_a (a_new - A)) + c_s S + U_C, so
    mean = c_t (T + b_a (a_new - A)) + c_s E[S | A, T] and
    var = c_s^2 Var[S | A, T] + sd_c^2."""
    # [m15c step 4]
    M, noise_sd = structural_matrix(scm)
    cov_x = M @ jnp.diag(noise_sd**2) @ M.T  # over (S, A, T, C)
    mu_x = jnp.zeros(4)
    A, T = jnp.asarray(obs["A"]), jnp.asarray(obs["T"])
    # condition the (S, A, T) block on A, T
    idx = jnp.array([0, 1, 2])
    cov3 = cov_x[jnp.ix_(idx, idx)]
    m_s, v_s = gaussian_condition(mu_x[idx], cov3, jnp.array([1, 2]), jnp.stack([A, T]))
    m_s, v_s = m_s[0], v_s[0, 0]
    mean = scm.c_t * (T + scm.b_a * (jnp.asarray(a_new) - A)) + scm.c_s * m_s
    sd = jnp.sqrt(scm.c_s**2 * v_s + scm.sd_c**2)
    return mean, sd


def counterfactual_posterior(
    scm: AdSpendSCM,
    obs: dict[str, Array],
    interventions: dict[str, Array],
    key: Array,
    method: str = "hmc",
    n_warmup: int = 300,
    n_samples: int = 300,
    n_chains: int = 4,
    n_svi_steps: int = 3000,
) -> dict[str, Array]:
    """Abduction by inference, action by `do`, prediction by running the program.

    1. Posterior samples of (S, U_C) from `abduction_model` via Module 13 `hmc` (default)
       or `svi`.
    2. For each draw, complete the noise with U_A = A - a_s S and U_T = T - b_s S - b_a A.
    3. Run do(ad_spend_scm, interventions) under substitute(noise) with jax.vmap.
    Returns every site of the counterfactual world with a leading sample axis."""
    # [m15c step 4]
    A, T = jnp.asarray(obs["A"]), jnp.asarray(obs["T"])
    k_inf, k_post = jax.random.split(key)
    if method == "hmc":
        samples, _, _ = hmc(abduction_model, (scm, obs), k_inf, n_warmup, n_samples, n_chains)
        S = samples["S"].reshape(-1)
        U_C = samples["U_C"].reshape(-1)
    elif method == "svi":
        params, _, spec = svi(abduction_model, (scm, obs), k_inf, n_svi_steps)
        ps = posterior_samples_svi(params, spec, k_post, n_chains * n_samples)
        S, U_C = ps["S"], ps["U_C"]
    else:
        raise ValueError(f"unknown method {method!r}")
    noise = {"S": S, "U_A": A - scm.a_s * S, "U_T": T - scm.b_s * S - scm.b_a * A, "U_C": U_C}
    model = do(ad_spend_scm, interventions)
    return jax.vmap(lambda u: run_with(model, u, scm))(noise)


# ======================================================================================
# Step 5: twin networks
# ======================================================================================
def twin_network(scm: AdSpendSCM, obs: dict[str, Array] | None = None) -> dict[str, Array]:
    """Factual and counterfactual worlds in one program, sharing the noise sites.

    Factual nodes A, T, C; counterfactual copies A_cf, T_cf, C_cf computed from the same
    S, U_A, U_T, U_C. Intervene on the counterfactual copy only: do(twin_network,
    {"A_cf": a}). With `obs` = {A, T}, the noise sites U_A and U_T are observed through
    their implied values as in `abduction_model`, so the latent sites are S and U_C and
    the factual A and T reproduce the observations exactly."""
    # [m15c step 5]
    S = sample("S", Normal(0.0, 1.0))
    if obs is None:
        U_A = sample("U_A", Normal(0.0, scm.sd_a))
        U_T = sample("U_T", Normal(0.0, scm.sd_t))
    else:
        A_obs, T_obs = jnp.asarray(obs["A"]), jnp.asarray(obs["T"])
        U_A = sample("U_A", Normal(0.0, scm.sd_a), obs=A_obs - scm.a_s * S)
        U_T = sample("U_T", Normal(0.0, scm.sd_t), obs=T_obs - scm.b_s * S - scm.b_a * A_obs)
    U_C = sample("U_C", Normal(0.0, scm.sd_c))
    A = deterministic("A", scm.a_s * S + U_A)
    T = deterministic("T", scm.b_s * S + scm.b_a * A + U_T)
    C = deterministic("C", scm.c_t * T + scm.c_s * S + U_C)
    A_cf = deterministic("A_cf", scm.a_s * S + U_A)
    T_cf = deterministic("T_cf", scm.b_s * S + scm.b_a * A_cf + U_T)
    C_cf = deterministic("C_cf", scm.c_t * T_cf + scm.c_s * S + U_C)
    return {"S": S, "A": A, "T": T, "C": C, "A_cf": A_cf, "T_cf": T_cf, "C_cf": C_cf}


def twin_posterior(scm: AdSpendSCM, obs: dict[str, Array], a_new: Array, key: Array, n_warmup: int = 300, n_samples: int = 300, n_chains: int = 4) -> dict[str, Array]:
    """Posterior samples of every twin-network site given obs = {A, T} and do(A_cf := a_new):
    Module 13 `hmc` on do(twin_network, {"A_cf": a_new}), then run the same program under
    substitute for each draw."""
    # [m15c step 5]
    model = do(twin_network, {"A_cf": a_new})
    samples, _, _ = hmc(model, (scm, obs), key, n_warmup, n_samples, n_chains)
    latents = {k: v.reshape(-1) for k, v in samples.items()}
    return jax.vmap(lambda u: run_with(model, u, scm, obs))(latents)


def probability_below(scm: AdSpendSCM, obs: dict[str, Array], a_new: Array, threshold: Array, key: Array, **kwargs) -> Array:
    """P(C_cf < threshold | A, T, do(A_cf := a_new)): the posterior probability that, had
    spend been a_new, conversions would have fallen below `threshold`."""
    # [m15c step 5]
    post = twin_posterior(scm, obs, a_new, key, **kwargs)
    return jnp.mean(post["C_cf"] < jnp.asarray(threshold))
