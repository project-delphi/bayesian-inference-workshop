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
    raise NotImplementedError  # Module 13, Step 1


def unflatten(z: Array, spec: ParamSpec) -> dict[str, Array]:
    """Flat unconstrained vector -> dict of constrained site values (apply each
    bijector's forward)."""
    raise NotImplementedError  # Module 13, Step 1


def flatten(values: dict[str, Array], spec: ParamSpec) -> Array:
    """Dict of constrained values -> flat unconstrained vector (apply each inverse)."""
    raise NotImplementedError  # Module 13, Step 1


# ======================================================================================
# Step 2: the model as a density on R^d
# ======================================================================================
def unconstrained_log_density(model: Callable, spec: ParamSpec, z: Array, *args, **kwargs) -> Array:
    """log p(T(z), data) + sum_sites log |det J_T(z)| where T applies the support
    bijectors. This is the function Modules 8 and 10 consume."""
    raise NotImplementedError  # Module 13, Step 2


# ======================================================================================
# Step 3: automatic mean-field VI
# ======================================================================================
def svi(model: Callable, args: tuple, key: Array, n_steps: int, lr: float = 0.02, n_samples: int = 8) -> tuple[dict[str, Array], Array, ParamSpec]:
    """Fit an AutoNormal guide (Module 8 MeanFieldGaussian on the unconstrained space)
    by maximising the ELBO with Module 8's `fit`. Returns (params, elbo_trace, spec)."""
    raise NotImplementedError  # Module 13, Step 3


def posterior_samples_svi(params: dict[str, Array], spec: ParamSpec, key: Array, n: int) -> dict[str, Array]:
    """Draw n samples from the fitted guide and map them to constrained site values;
    each entry has leading dimension n."""
    raise NotImplementedError  # Module 13, Step 3


# ======================================================================================
# Step 4: HMC through the PPL
# ======================================================================================
def init_from_prior(model: Callable, args: tuple, spec: ParamSpec, key: Array, n_chains: int) -> Array:
    """One unconstrained starting point per chain, drawn from the prior by running the
    model under seed and flattening the latent values."""
    raise NotImplementedError  # Module 13, Step 4


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
    raise NotImplementedError  # Module 13, Step 4


# ======================================================================================
# Step 5: posterior predictive
# ======================================================================================
def predictive(model: Callable, posterior_samples: dict[str, Array], key: Array, *args, **kwargs) -> dict[str, Array]:
    """For each posterior draw (leading axis of every entry), run the model with its
    latent sites substituted, then draw fresh values at the observed sites from their
    distributions evaluated at that draw. Returns a dict of observed-site samples with
    the same leading axis. Uses jax.vmap over draws with one key per draw."""
    raise NotImplementedError  # Module 13, Step 5


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
