import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate

from workshop.m02b_foundations import grid_log_posterior, log_joint, posterior_predictive_grid
from tests.m02b._common import LOGLAM_GRID, MU_GRID, X, np_log_joint_constrained


def test_step3_predictive_integrates_to_one_and_matches_quadrature():
    log_post, log_ev = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    xs = jnp.linspace(-1.0, 4.5, 1101)
    lp = jax.vmap(lambda xn: posterior_predictive_grid(log_post, MU_GRID, LOGLAM_GRID, xn))(xs)
    np.testing.assert_allclose(jnp.trapezoid(jnp.exp(lp), xs), 1.0, atol=2e-3)
    shift = float(np_log_joint_constrained(1.8, 9.0))
    for x_new in (1.5, 2.3):
        def integrand(lam, mu):
            lik_new = np.sqrt(lam / (2 * np.pi)) * np.exp(-0.5 * lam * (x_new - mu) ** 2)
            return np.exp(np_log_joint_constrained(mu, lam) - shift) * lik_new
        num, _ = scipy.integrate.dblquad(integrand, 1.0, 2.6, 0.5, 60.0)
        ref = np.log(num) + shift - float(log_ev)
        np.testing.assert_allclose(posterior_predictive_grid(log_post, MU_GRID, LOGLAM_GRID, jnp.asarray(x_new)), ref, atol=3e-3)


def test_step3_predictive_is_wider_than_plug_in():
    log_post, _ = grid_log_posterior(log_joint, X, MU_GRID, LOGLAM_GRID)
    xs = jnp.linspace(-1.0, 4.5, 1101)
    lp = jax.vmap(lambda xn: posterior_predictive_grid(log_post, MU_GRID, LOGLAM_GRID, xn))(xs)
    p = jnp.exp(lp)
    m = jnp.trapezoid(p * xs, xs)
    v = jnp.trapezoid(p * (xs - m) ** 2, xs)
    assert float(v) > float(X.var(ddof=1)) * 0.95  # predictive variance >= noise variance (approximately)
