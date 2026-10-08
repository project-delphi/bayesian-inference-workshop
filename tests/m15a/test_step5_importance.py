import jax
import jax.numpy as jnp
import numpy as np

from workshop.m08_bbvi import MeanFieldGaussian
from workshop.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob
from workshop.m15a_flows import flow_sample, importance_ess, importance_weighted_elbo
from tests._util import key
from tests.m15a._common import flow_fit, mean_field_fit


def test_step5_ess_of_uniform_weights_is_n():
    np.testing.assert_allclose(importance_ess(jnp.full(500, 3.0)), 500.0, rtol=1e-10)
    lw = jnp.array([0.0, -jnp.inf, -jnp.inf])
    np.testing.assert_allclose(importance_ess(lw), 1.0, rtol=1e-10)


def test_step5_bound_tightens_with_k():
    fp, _ = flow_fit()
    bounds = [float(importance_weighted_elbo(uplift_centered_log_prob, fp, key(1520), 2000, k)) for k in (1, 4, 16, 64)]
    for lo, hi in zip(bounds, bounds[1:]):
        assert hi > lo - 0.03, bounds
    assert bounds[-1] > bounds[0] + 0.1, bounds


def test_step5_flow_proposal_has_far_higher_ess_than_mean_field():
    fp, _ = flow_fit()
    mp, _ = mean_field_fit()
    n = 20_000
    xf, lqf = flow_sample(key(1521), fp, n)
    lw_f = jax.vmap(uplift_centered_log_prob)(xf) - lqf
    mf = MeanFieldGaussian(UPLIFT_DIM)
    xm = mf.sample(key(1522), mp, n)
    lw_m = jax.vmap(uplift_centered_log_prob)(xm) - jax.vmap(lambda z: mf.log_prob(mp, z))(xm)
    ess_f, ess_m = float(importance_ess(lw_f)) / n, float(importance_ess(lw_m)) / n
    assert ess_f > 0.1, ess_f
    assert ess_f > 10 * ess_m, (ess_f, ess_m)
