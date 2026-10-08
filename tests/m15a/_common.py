"""Cached fits shared by the Module 15a tests (steps 4 and 5)."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from tests._util import key

N_LAYERS, HIDDEN, N_STEPS, LR, N_SAMPLES = 4, 32, 12_000, 2e-3, 64


@functools.lru_cache(maxsize=None)
def flow_fit():
    from workshop.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob
    from workshop.m15a_flows import fit_flow_vi, init_flow

    params = init_flow(key(1501), UPLIFT_DIM, N_LAYERS, HIDDEN)
    return fit_flow_vi(uplift_centered_log_prob, params, key(1502), N_STEPS, lr=LR, n_samples=N_SAMPLES)


@functools.lru_cache(maxsize=None)
def mean_field_fit():
    from workshop.m08_bbvi import fit
    from workshop.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob

    return fit(uplift_centered_log_prob, UPLIFT_DIM, key(1503), 4000, lr=0.01, n_samples=16)


@functools.lru_cache(maxsize=None)
def reference():
    from workshop.m15a_flows import hmc_reference

    return hmc_reference(key(1504))
