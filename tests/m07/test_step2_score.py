import jax
import jax.numpy as jnp
import numpy as np

from workshop.m07_gradients import elbo_grad_reparam, elbo_grad_score, laplace
from tests._util import key
from tests.m07._common import log_joint


def _params():
    mean, cov = laplace(log_joint, jnp.zeros(5))
    return {"loc": mean + 0.3, "log_scale": jnp.log(jnp.sqrt(jnp.diag(cov))) + 0.2}


def test_step2_score_estimator_is_unbiased():
    params = _params()
    n, R = 2000, 60
    keys = jax.random.split(key(300), R)
    flat = lambda g: jnp.concatenate([g["loc"], g["log_scale"]])
    S = jax.vmap(lambda k: flat(elbo_grad_score(k, params, log_joint, n)))(keys)
    ref = flat(elbo_grad_reparam(key(301), params, log_joint, 200_000))
    se = S.std(0) / np.sqrt(R)
    gap = jnp.abs(S.mean(0) - ref)
    assert bool(jnp.all(gap < 5 * se + 1e-3)), (gap, se)


def test_step2_output_structure():
    params = _params()
    g = elbo_grad_score(key(302), params, log_joint, 16)
    assert set(g) == {"loc", "log_scale"}
    assert g["loc"].shape == (5,) and g["log_scale"].shape == (5,)
    assert bool(jnp.all(jnp.isfinite(g["loc"])))
