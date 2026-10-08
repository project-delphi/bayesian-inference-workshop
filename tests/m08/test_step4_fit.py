import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import saas_churn
from workshop.m07_gradients import churn_log_joint, laplace, laplace_log_evidence
from workshop.m08_bbvi import MeanFieldGaussian, fit
from tests._util import key


def test_step4_exact_on_diagonal_gaussian_target():
    m, s = jnp.array([1.0, -2.0, 0.3]), jnp.array([0.5, 2.0, 1.0])
    lj = lambda z: jnp.sum(-0.5 * ((z - m) / s) ** 2 - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi))
    params, trace = fit(lj, 3, key(420), n_steps=4000, lr=0.01, n_samples=16)
    assert bool(jnp.all(jnp.abs(params["loc"] - m) < 0.06 * s)), (params["loc"] - m) / s
    np.testing.assert_allclose(jnp.exp(params["log_scale"]), s, rtol=0.08)
    assert trace.shape == (4000,)
    assert abs(float(trace[-200:].mean())) < 0.1  # ELBO -> log evidence = 0 for a normalised target


def test_step4_churn_posterior_matches_laplace():
    X, y, _ = (jnp.asarray(a, dtype=float) for a in saas_churn())
    lj = lambda b: churn_log_joint(b, X, y)
    mean, cov = laplace(lj, jnp.zeros(5))
    sd = jnp.sqrt(jnp.diag(cov))
    params, trace = fit(lj, 5, key(421), n_steps=2500, lr=0.02, n_samples=8)
    assert bool(jnp.all(jnp.abs(params["loc"] - mean) < 3 * sd))
    assert bool(jnp.all(jnp.exp(params["log_scale"]) < 1.5 * sd))  # mean-field under-dispersion
    assert float(trace[-100:].mean()) > float(trace[:100].mean())
    ev = laplace_log_evidence(lj, mean, cov)
    final = float(trace[-300:].mean())
    assert ev - 3.0 < final < ev + 1.0, (final, ev)
