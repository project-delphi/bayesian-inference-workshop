import jax
import jax.numpy as jnp
import numpy as np

from workshop.m13_ppl_inference import enzyme_model, posterior_samples_svi, svi
from tests.m13._common import KM_TRUE, S, V, VMAX_TRUE
from tests._util import key


def test_step3_svi_recovers_kinetic_constants():
    params, elbo, spec = svi(enzyme_model, (S, V), key(50), n_steps=6000, lr=0.02, n_samples=32)
    assert elbo.shape == (6000,)
    assert elbo[-200:].mean() > elbo[:200].mean()
    ps = posterior_samples_svi(params, spec, key(51), 4000)
    assert ps["Vmax"].shape == (4000,)
    assert abs(float(ps["Vmax"].mean()) - VMAX_TRUE) < 1.0
    assert abs(float(ps["Km"].mean()) - KM_TRUE) < 0.6
    assert 0.3 < float(ps["sigma"].mean()) < 1.0
    assert bool(jnp.all(ps["sigma"] > 0))
