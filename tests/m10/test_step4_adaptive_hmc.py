import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import saas_churn
from workshop.m10_adaptation import adaptive_hmc, hmc_jittered
from tests._util import key


def _churn():
    X, y, _ = saas_churn()
    X, y = jnp.asarray(X), jnp.asarray(y, dtype=float)

    def lj(b):
        logits = X @ b
        return jnp.sum(y * logits - jax.nn.softplus(logits)) - 0.5 * jnp.sum((b / 2.5) ** 2)

    return lj


def test_step4_jitter_uses_variable_trajectory_lengths():
    lp = lambda z: -0.5 * jnp.sum(z**2)
    s, info = hmc_jittered(lp, key(260), jnp.zeros(3), 2000, jnp.asarray(0.3), 12, jnp.ones(3))
    assert s.shape == (2000, 3)
    # with varying L the energy errors are not all identical in magnitude pattern
    assert float(jnp.std(info.energy_error)) > 0
    np.testing.assert_allclose(s.var(0), 1.0, atol=0.15)


def test_step4_churn_posterior_matches_laplace():
    lj = _churn()
    g, H = jax.grad(lj), jax.hessian(lj)
    mode = jax.lax.fori_loop(0, 30, lambda i, b: b - jnp.linalg.solve(H(b), g(b)), jnp.zeros(5))
    sd = jnp.sqrt(jnp.diag(jnp.linalg.inv(-H(mode))))
    samples, info, adapted = jax.jit(lambda k: adaptive_hmc(lj, k, jnp.zeros(5), 500, 1500))(key(261))
    assert samples.shape == (1500, 5)
    assert float(info.accept_prob.mean()) > 0.6
    assert bool(jnp.all(jnp.abs(samples.mean(0) - mode) <= 0.15 * sd + 0.02))
    np.testing.assert_allclose(samples.std(0), sd, rtol=0.25)
    assert bool(jnp.all(adapted.inv_mass > 0))
