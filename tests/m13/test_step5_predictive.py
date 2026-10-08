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


def test_step5_predictive_draws_at_the_observation_shape():
    # A scalar Normal observing a length-400 vector: each posterior draw must yield 400
    # fresh replicates, not one scalar.
    from workshop.m12_handlers import two_site_model
    from workshop.m13_ppl_inference import predictive

    mu = jnp.arange(5.0)
    out = predictive(two_site_model, {"mu": mu}, key(55), jnp.zeros(400))
    assert out["x"].shape == (5, 400)
    np.testing.assert_allclose(out["x"].mean(axis=1), mu, atol=0.15)
    np.testing.assert_allclose(out["x"].std(axis=1), 0.5, atol=0.08)


def test_step5_predictive_broadcasts_size_one_batch_dimensions():
    # Batch shape (5, 1) observing a (5, 3) array: every one of the 15 entries is drawn.
    from workshop.m12_handlers import Normal, sample
    from workshop.m13_ppl_inference import predictive

    def model(y):
        mu = sample("mu", Normal(jnp.zeros(5), 1.0))
        sample("y", Normal(mu[:, None], 0.1), obs=y)

    draws = {"mu": jnp.tile(jnp.arange(5.0), (4, 1))}
    out = predictive(model, draws, key(56), jnp.zeros((5, 3)))
    assert out["y"].shape == (4, 5, 3)
    assert float(jnp.std(out["y"][0], axis=1).min()) > 0.0
