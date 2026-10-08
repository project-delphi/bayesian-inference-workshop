import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import hmc, hmc_step, standard_normal_log_prob
from workshop.m10_adaptation import dual_averaging_final, dual_averaging_init, dual_averaging_update
from tests._util import key


def test_step1_update_formulas():
    s = dual_averaging_init(0.1)
    np.testing.assert_allclose(s.mu, np.log(1.0))
    s1 = dual_averaging_update(s, jnp.asarray(0.5))
    # t=1: h_bar = (0.8-0.5)/11 ; log_step = mu - 1/0.05 * h_bar ; avg = log_step
    h = 0.3 / 11
    np.testing.assert_allclose(s1.h_bar, h)
    np.testing.assert_allclose(s1.log_step, np.log(1.0) - np.sqrt(1) / 0.05 * h)
    np.testing.assert_allclose(s1.log_step_avg, s1.log_step)
    assert float(s1.t) == 1.0


def test_step1_adapts_to_target_acceptance():
    target = 0.8
    dim = 5
    lp = lambda z: -0.5 * jnp.sum(z**2)

    def body(carry, k):
        q, da = carry
        q, info = hmc_step(lp, k, q, jnp.exp(da.log_step), 10, jnp.ones(dim))
        da = dual_averaging_update(da, info.accept_prob, target=target)
        return (q, da), info.accept_prob

    keys = jax.random.split(key(230), 1500)
    (q, da), _ = jax.lax.scan(body, (jnp.zeros(dim), dual_averaging_init(1.0)), keys)
    eps = dual_averaging_final(da)
    assert 0.05 < float(eps) < 2.0
    _, info = hmc(lp, key(231), q, 3000, float(eps), 10)
    assert abs(float(info.accept_prob.mean()) - target) < 0.1
