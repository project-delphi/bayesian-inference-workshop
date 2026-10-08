import time

import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import ad_clicks
from workshop.m07_gradients import laplace
from workshop.m08_bbvi import fit_svi
from tests._util import key


def _parts():
    X, y, beta = (jnp.asarray(a, dtype=float) for a in ad_clicks())
    log_prior = lambda b: -0.5 * jnp.sum(b**2) / 2.5**2
    log_lik = lambda b, Xb, yb: jnp.sum(yb * jax.nn.log_sigmoid(Xb @ b) + (1 - yb) * jax.nn.log_sigmoid(-(Xb @ b)))
    return X, y, beta, log_prior, log_lik


def test_step5_svi_matches_full_data_laplace():
    X, y, beta, log_prior, log_lik = _parts()
    mean, cov = laplace(lambda b: log_prior(b) + log_lik(b, X, y), jnp.zeros(9))
    sd = jnp.sqrt(jnp.diag(cov))
    t0 = time.time()
    params, trace = fit_svi(log_prior, log_lik, (X, y), 9, key(430), n_steps=3000, batch_size=500, lr=0.02, n_samples=4)
    elapsed = time.time() - t0
    assert elapsed < 30, elapsed
    assert trace.shape == (3000,)
    assert bool(jnp.all(jnp.abs(params["loc"] - mean) < 4 * sd + 0.02)), (params["loc"] - mean) / sd
    assert float(jnp.abs(params["loc"] - beta).max()) < 0.2  # true coefficients, sanity
    assert float(trace[-200:].mean()) > float(trace[:200].mean())
