import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from workshop.data import saas_churn
from workshop.m11_diagnostics import churn_log_joint, compare_hmc_vs_vi, laplace_approximation, run_chains, summarize
from tests._util import key


def _setup():
    X, y, _ = saas_churn()
    X, y = jnp.asarray(X), jnp.asarray(y, dtype=float)
    lj = lambda b: churn_log_joint(b, X, y)
    chains, info, adapted = run_chains(lj, key(300), 0.1 * jax.random.normal(key(301), (4, 5)), 500, 500)
    return lj, chains, info


def test_step4_summary_matches_laplace():
    lj, chains, info = _setup()
    s = summarize(chains)
    mode, cov = laplace_approximation(lj, jnp.zeros(5))
    sd = jnp.sqrt(jnp.diag(cov))
    assert set(s) >= {"mean", "sd", "q05", "q95", "rhat", "ess"}
    assert bool(jnp.all(jnp.abs(s["mean"] - mode) <= 0.1 * sd + 0.02))
    assert bool(jnp.all(s["rhat"] < 1.05))
    assert float(s["ess"].min()) > 400
    assert bool(jnp.all(s["q05"] < s["mean"])) and bool(jnp.all(s["q95"] > s["mean"]))


def test_step4_compare_structure_with_laplace_as_stand_in():
    lj, chains, _ = _setup()
    mode, cov = laplace_approximation(lj, jnp.zeros(5))
    out = compare_hmc_vs_vi(chains, mode, jnp.sqrt(jnp.diag(cov)))
    assert out["mean_diff_in_sd"].shape == (5,) and out["sd_ratio"].shape == (5,)
    assert float(jnp.abs(out["mean_diff_in_sd"]).max()) < 0.15
    assert 0 <= float(out["max_abs_corr"]) <= 1
    # the churn posterior has a non-trivial correlation structure that mean-field ignores
    assert float(out["max_abs_corr"]) > 0.1


def test_step4_mean_field_vi_is_underdispersed():
    try:
        m08 = importlib.import_module("workshop.m08_bbvi")
        fit, MeanFieldGaussian = m08.fit, m08.MeanFieldGaussian
    except (ImportError, AttributeError):
        pytest.skip("Module 8 (black-box VI engine) is not available yet")
    lj, chains, _ = _setup()
    params, elbo = fit(lj, 5, key(302), n_steps=3000, lr=0.02, n_samples=8)
    out = compare_hmc_vs_vi(chains, params["loc"], jnp.exp(params["log_scale"]))
    assert float(jnp.abs(out["mean_diff_in_sd"]).max()) < 0.3
    assert float(out["sd_ratio"].mean()) <= 1.0
    assert bool(jnp.all(out["sd_ratio"] < 1.1))
