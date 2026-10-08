import jax
import jax.numpy as jnp
import numpy as np

from workshop.m11_diagnostics import divergences
from workshop.m12_handlers import uplift_model
from workshop.m13_ppl_inference import enzyme_model, hmc, posterior_samples_svi, svi
from tests.m13._common import KM_TRUE, S, SIGMA, V, VMAX_TRUE, Y
from tests._util import key


def test_step4_hmc_on_enzyme_model_mixes_and_agrees_with_svi():
    samples, infos, summary = hmc(enzyme_model, (S, V), key(60), n_warmup=300, n_samples=300, n_chains=4)
    assert samples["Vmax"].shape == (4, 300)
    assert summary["names"] == ["Vmax", "Km", "sigma"]
    assert bool(jnp.all(summary["rhat"] < 1.05))
    assert bool(jnp.all(summary["ess"] > 200))
    assert int(divergences(infos)) <= 5
    vmax, km = samples["Vmax"].reshape(-1), samples["Km"].reshape(-1)
    assert abs(float(vmax.mean()) - VMAX_TRUE) < 1.0 and abs(float(km.mean()) - KM_TRUE) < 0.6
    params, _, spec = svi(enzyme_model, (S, V), key(61), n_steps=6000, lr=0.02, n_samples=32)
    ps = posterior_samples_svi(params, spec, key(62), 4000)
    for name in ("Vmax", "Km", "sigma"):
        h = samples[name].reshape(-1)
        assert abs(float(ps[name].mean()) - float(h.mean())) < 3 * float(h.std()), name
    # mean-field under-dispersion on the correlated (Vmax, Km) pair
    assert float(ps["Vmax"].std()) < float(vmax.std())


def test_step4_noncentred_uplift_model_runs_without_divergences():
    samples, infos, summary = hmc(uplift_model, (Y, SIGMA), key(63), n_warmup=400, n_samples=400, n_chains=4)
    assert samples["eta"].shape == (4, 400, 8)
    assert int(divergences(infos)) <= 5
    assert bool(jnp.all(samples["tau"] > 0))
    assert bool(jnp.all(summary["rhat"] < 1.1))
