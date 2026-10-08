import functools

import jax
import jax.numpy as jnp
import numpy as np

from workshop.m15a_flows import fit_flow_vi, flow_elbo, flow_sample, init_flow
from tests._util import key

COV = jnp.array([[1.0, 0.8], [0.8, 1.0]])
MU = jnp.array([1.0, -2.0])
_LINV = jnp.linalg.inv(jnp.linalg.cholesky(COV))


def target(x):
    return -0.5 * jnp.sum((_LINV @ (x - MU)) ** 2) - jnp.log(2 * jnp.pi) - 0.5 * jnp.linalg.slogdet(COV)[1]


@functools.lru_cache(maxsize=None)
def _fit():
    p = init_flow(key(30), 2, 4)
    return fit_flow_vi(target, p, key(31), 3000, lr=3e-3, n_samples=32)


def test_step3_recovers_mean_and_covariance():
    p, _ = _fit()
    x, _ = flow_sample(key(32), p, 20_000)
    np.testing.assert_allclose(x.mean(0), MU, atol=0.1)
    np.testing.assert_allclose(jnp.cov(x.T), COV, atol=0.1)


def test_step3_elbo_near_log_normaliser():
    p, _ = _fit()
    elbo = flow_elbo(key(33), p, target, 20_000)
    assert -0.05 < float(elbo) <= 0.02, float(elbo)


def test_step3_trace_increases():
    _, trace = _fit()
    assert trace.shape == (3000,)
    assert bool(jnp.all(jnp.isfinite(trace)))
    assert float(trace[-500:].mean()) > float(trace[:100].mean()) + 0.5
