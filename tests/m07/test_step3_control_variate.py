import jax.numpy as jnp

from workshop.m07_gradients import elbo_grad_reparam, elbo_grad_score, elbo_grad_score_cv, estimator_variance, laplace
from tests._util import key
from tests.m07._common import log_joint
import jax, numpy as np


def _params():
    mean, cov = laplace(log_joint, jnp.zeros(5))
    return {"loc": mean + 0.3, "log_scale": jnp.log(jnp.sqrt(jnp.diag(cov))) + 0.2}


def test_step3_control_variate_reduces_variance():
    params = _params()
    v_plain = estimator_variance(key(310), elbo_grad_score, params, log_joint, 64, 300)
    v_cv = estimator_variance(key(310), elbo_grad_score_cv, params, log_joint, 64, 300)
    assert bool(jnp.all(v_cv < v_plain))
    assert float(v_cv.sum()) < 0.5 * float(v_plain.sum())


def test_step3_control_variate_remains_nearly_unbiased():
    params = _params()
    n, R = 2000, 60
    keys = jax.random.split(key(311), R)
    flat = lambda g: jnp.concatenate([g["loc"], g["log_scale"]])
    S = jax.vmap(lambda k: flat(elbo_grad_score_cv(k, params, log_joint, n)))(keys)
    ref = flat(elbo_grad_reparam(key(312), params, log_joint, 200_000))
    se = S.std(0) / np.sqrt(R)
    assert bool(jnp.all(jnp.abs(S.mean(0) - ref) < 5 * se + 1e-2))
