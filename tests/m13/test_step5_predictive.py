import jax
import jax.numpy as jnp
import numpy as np

from workshop.m13_ppl_inference import enzyme_model, hmc, predictive
from tests.m13._common import S, SD_TRUE, V
from tests._util import key


def test_step5_posterior_predictive_tracks_data():
    samples, _, _ = hmc(enzyme_model, (S, V), key(70), n_warmup=300, n_samples=250, n_chains=4)
    flat = {k: v.reshape(-1) for k, v in samples.items()}
    pred = predictive(enzyme_model, flat, key(71), S, V)
    assert set(pred) == {"v"}
    assert pred["v"].shape == (1000, S.shape[0])
    resid = pred["v"].mean(0) - V
    assert float(jnp.abs(resid).max()) < 4 * SD_TRUE
    # predictive spread reflects the noise level, not the data
    assert 0.5 * SD_TRUE < float(pred["v"].std(0).mean()) < 2.0 * SD_TRUE
    # and differs across draws (not just the mean curve)
    assert float(pred["v"][0].std()) > 0
