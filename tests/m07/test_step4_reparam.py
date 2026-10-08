import jax
import jax.numpy as jnp
import numpy as np

from workshop.m07_gradients import elbo_grad_reparam, elbo_grad_score, estimator_variance, guide_log_prob, laplace
from tests._util import key
from tests.m07._common import log_joint


def _params():
    mean, cov = laplace(log_joint, jnp.zeros(5))
    return {"loc": mean + 0.3, "log_scale": jnp.log(jnp.sqrt(jnp.diag(cov))) + 0.2}


def test_step4_reparam_exact_for_gaussian_target():
    # If log_joint is N(m, s^2 I) and q is N(loc, scale^2 I), the ELBO gradient is known.
    m, s = jnp.array([1.0, -2.0]), jnp.array([0.5, 2.0])
    lj = lambda z: jnp.sum(-0.5 * ((z - m) / s) ** 2 - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi))
    params = {"loc": jnp.array([0.0, 0.0]), "log_scale": jnp.array([0.0, 0.0])}
    g = elbo_grad_reparam(key(320), params, lj, 400_000)
    # d/dloc = -(loc - m)/s^2 ; d/dlog_scale = -scale^2/s^2 + 1
    np.testing.assert_allclose(g["loc"], -(params["loc"] - m) / s**2, atol=2e-2)
    np.testing.assert_allclose(g["log_scale"], 1 - jnp.exp(2 * params["log_scale"]) / s**2, atol=2e-2)


def test_step4_reparam_variance_much_lower_than_score():
    params = _params()
    v_score = estimator_variance(key(321), elbo_grad_score, params, log_joint, 64, 300).sum()
    v_rep = estimator_variance(key(321), elbo_grad_reparam, params, log_joint, 64, 300).sum()
    assert float(v_rep) * 10 < float(v_score), (v_rep, v_score)
