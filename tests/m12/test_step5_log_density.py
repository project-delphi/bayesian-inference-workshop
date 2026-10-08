import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import regional_uplift
from workshop.m09_hmc import uplift_noncentered_log_prob
from workshop.m12_handlers import log_density, uplift_model
from tests._util import key

Y, SIGMA = (jnp.asarray(a) for a in regional_uplift())


def _values(z):
    return {"mu": z[0], "tau": jnp.exp(z[1]), "eta": z[2:]}


def _ppl_lp(z):
    # Module 9 works in log tau and includes the Jacobian +log tau; the PPL works in tau.
    return log_density(uplift_model, _values(z), Y, SIGMA)[0] + z[1]


def test_step5_matches_module9_target_up_to_a_constant():
    z1 = jax.random.normal(key(30), (10,))
    z2 = jax.random.normal(key(31), (10,))
    # Module 9 drops the Gaussian normalising constants; differences must agree exactly.
    np.testing.assert_allclose(_ppl_lp(z1) - _ppl_lp(z2), uplift_noncentered_log_prob(z1) - uplift_noncentered_log_prob(z2), rtol=1e-10)


def test_step5_gradient_matches_module9_and_is_jittable():
    z = jax.random.normal(key(32), (10,))
    g_ppl = jax.jit(jax.grad(_ppl_lp))(z)
    g_ref = jax.grad(uplift_noncentered_log_prob)(z)
    np.testing.assert_allclose(g_ppl, g_ref, rtol=1e-10, atol=1e-12)


def test_step5_grad_with_respect_to_values_dict():
    z = jax.random.normal(key(33), (10,))
    vals = _values(z)
    g = jax.grad(lambda v: log_density(uplift_model, v, Y, SIGMA)[0])(vals)
    assert set(g) == {"mu", "tau", "eta"} and g["eta"].shape == (8,)
    # d/d tau of the Gaussian likelihood term by hand: sum_j eta_j (y_j - theta_j)/sigma_j^2 - tau/25
    theta = vals["mu"] + vals["tau"] * vals["eta"]
    ref = jnp.sum(vals["eta"] * (Y - theta) / SIGMA**2) - vals["tau"] / 25.0
    np.testing.assert_allclose(g["tau"], ref, rtol=1e-10)
