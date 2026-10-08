"""Module 13 — Inference through the PPL.

Given a model written with the Module 12 primitives, this module (1) discovers its
latent sites by tracing one execution and lays them out in a flat unconstrained vector
with each site's support bijector, (2) exposes the model as a function R^d -> R (log
density plus log-Jacobian), and (3) runs the Module 8 variational engine and the Module
10/11 Hamiltonian sampler on that function, mapping results back to named, constrained
samples. Nothing in Modules 8 to 11 changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .m08_bbvi import Bijector, MeanFieldGaussian, fit
from .m11_diagnostics import run_chains, summarize
from .m12_handlers import HalfNormal, LogNormal, Normal, log_density, sample, seed, substitute, trace

Array = jax.Array


# ======================================================================================
# Step 1: parameter layout
# ======================================================================================
@dataclass(frozen=True)
class ParamSpec:
    """Layout of the latent sites of a model in one flat unconstrained vector."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    slices: tuple[slice, ...]
    bijectors: tuple[Bijector, ...]
    dim: int


def unconstrain(model: Callable, *args, **kwargs) -> ParamSpec:
    """Trace one execution of `model` (seeded with a fixed key) and build a ParamSpec from
    its latent sample sites, in execution order. A latent site whose distribution has
    `support = None` (discrete) is an error."""
    # [m13 step 1]
    tr = trace(seed(model, jax.random.key(0))).get_trace(*args, **kwargs)
    names, shapes, slices, bijectors = [], [], [], []
    offset = 0
    for name, site in tr.items():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        bij = site["fn"].support
        if bij is None:
            raise ValueError(f"latent site '{name}' is discrete; marginalise it or use a continuous relaxation")
        shape = tuple(jnp.shape(site["value"]))
        size = 1
        for d in shape:
            size *= int(d)
        names.append(name)
        shapes.append(shape)
        slices.append(slice(offset, offset + size))
        bijectors.append(bij)
        offset += size
    return ParamSpec(tuple(names), tuple(shapes), tuple(slices), tuple(bijectors), offset)


def unflatten(z: Array, spec: ParamSpec) -> dict[str, Array]:
    """Flat unconstrained vector -> dict of constrained site values (apply each
    bijector's forward)."""
    # [m13 step 1]
    out = {}
    for name, shape, sl, bij in zip(spec.names, spec.shapes, spec.slices, spec.bijectors):
        out[name] = bij.forward(z[sl].reshape(shape))
    return out


def flatten(values: dict[str, Array], spec: ParamSpec) -> Array:
    """Dict of constrained values -> flat unconstrained vector (apply each inverse)."""
    # [m13 step 1]
    parts = [spec.bijectors[i].inverse(jnp.asarray(values[n])).reshape(-1) for i, n in enumerate(spec.names)]
    return jnp.concatenate(parts) if parts else jnp.zeros(0)


# ======================================================================================
# Step 2: the model as a density on R^d
# ======================================================================================
def unconstrained_log_density(model: Callable, spec: ParamSpec, z: Array, *args, **kwargs) -> Array:
    """log p(T(z), data) + sum_sites log |det J_T(z)| where T applies the support
    bijectors. This is the function Modules 8 and 10 consume."""
    # [m13 step 2]
    values = unflatten(z, spec)
    lp, _ = log_density(model, values, *args, **kwargs)
    for shape, sl, bij in zip(spec.shapes, spec.slices, spec.bijectors):
        lp = lp + bij.log_det_jacobian(z[sl].reshape(shape))
    return lp


# ======================================================================================
# Step 3: automatic mean-field VI
# ======================================================================================
def svi(model: Callable, args: tuple, key: Array, n_steps: int, lr: float = 0.02, n_samples: int = 8) -> tuple[dict[str, Array], Array, ParamSpec]:
    """Fit an AutoNormal guide (Module 8 MeanFieldGaussian on the unconstrained space)
    by maximising the ELBO with Module 8's `fit`. Returns (params, elbo_trace, spec)."""
    # [m13 step 3]
    spec = unconstrain(model, *args)
    lj = lambda z: unconstrained_log_density(model, spec, z, *args)
    params, elbo = fit(lj, spec.dim, key, n_steps, lr=lr, n_samples=n_samples)
    return params, elbo, spec


def posterior_samples_svi(params: dict[str, Array], spec: ParamSpec, key: Array, n: int) -> dict[str, Array]:
    """Draw n samples from the fitted guide and map them to constrained site values;
    each entry has leading dimension n."""
    # [m13 step 3]
    z = MeanFieldGaussian(spec.dim).sample(key, params, n)
    return jax.vmap(lambda row: unflatten(row, spec))(z)


# ======================================================================================
# Step 4: HMC through the PPL
# ======================================================================================
def init_from_prior(model: Callable, args: tuple, spec: ParamSpec, key: Array, n_chains: int) -> Array:
    """One unconstrained starting point per chain, drawn from the prior by running the
    model under seed and flattening the latent values."""
    # [m13 step 4]

    def one(k):
        tr = trace(seed(model, k)).get_trace(*args)
        vals = {n: tr[n]["value"] for n in spec.names}
        return flatten(vals, spec)

    return jnp.stack([one(k) for k in jax.random.split(key, n_chains)])


def flat_names(spec: ParamSpec) -> list[str]:
    """Human-readable name of each coordinate of the flat vector."""
    out = []
    for name, shape in zip(spec.names, spec.shapes):
        if shape == ():
            out.append(name)
        else:
            n = 1
            for d in shape:
                n *= int(d)
            out.extend(f"{name}[{i}]" for i in range(n))
    return out


def hmc(model: Callable, args: tuple, key: Array, n_warmup: int, n_samples: int, n_chains: int = 4, **adapt_kwargs) -> tuple[dict[str, Array], Any, dict[str, Array]]:
    """Adaptive HMC (Module 11 `run_chains`, which wraps Module 10 `adaptive_hmc`) on
    the unconstrained density, started from prior draws.

    Returns (samples: dict name -> array [n_chains, n_samples, *shape] in constrained
    space, infos: HMCInfo [n_chains, n_samples], summary: Module 11 `summarize` of the
    unconstrained chains plus a "names" entry listing the flat coordinates)."""
    # [m13 step 4]
    spec = unconstrain(model, *args)
    k_init, k_run = jax.random.split(key)
    q0s = init_from_prior(model, args, spec, k_init, n_chains)
    lj = lambda z: unconstrained_log_density(model, spec, z, *args)
    chains, infos, _ = run_chains(lj, k_run, q0s, n_warmup, n_samples, **adapt_kwargs)
    samples = jax.vmap(jax.vmap(lambda row: unflatten(row, spec)))(chains)
    summary = summarize(chains)
    summary["names"] = flat_names(spec)
    return samples, infos, summary


# ======================================================================================
# Step 5: posterior predictive
# ======================================================================================
def predictive(model: Callable, posterior_samples: dict[str, Array], key: Array, *args, **kwargs) -> dict[str, Array]:
    """For each posterior draw (leading axis of every entry), run the model with its
    latent sites substituted, then draw fresh values at the observed sites from their
    distributions evaluated at that draw. Returns a dict of observed-site samples with
    the same leading axis. Uses jax.vmap over draws with one key per draw."""
    # [m13 step 5]
    n = jax.tree_util.tree_leaves(posterior_samples)[0].shape[0]
    probe = trace(seed(model, jax.random.key(0))).get_trace(*args, **kwargs)
    observed = [name for name, s in probe.items() if s["type"] == "sample" and s["is_observed"]]

    def one(k, draw):
        tr = trace(substitute(model, draw)).get_trace(*args, **kwargs)
        return {name: tr[name]["fn"].sample(jax.random.fold_in(k, i)) for i, name in enumerate(observed)}

    return jax.vmap(one)(jax.random.split(key, n), posterior_samples)


# ======================================================================================
# Example model
# ======================================================================================
def enzyme_model(s: Array, v: Array) -> None:
    """Michaelis–Menten kinetics: v = Vmax s / (Km + s) + noise, for data.enzyme_kinetics()."""
    vmax = sample("Vmax", LogNormal(jnp.log(10.0), 1.0))
    km = sample("Km", LogNormal(0.0, 1.0))
    sigma = sample("sigma", HalfNormal(2.0))
    mean = vmax * s / (km + s)
    sample("v", Normal(mean, sigma), obs=v)
