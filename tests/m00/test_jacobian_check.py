"""The change-of-variables checker is a provided tool, so it is tested here in m00
(exempt from the "must fail on the starter" rule) rather than in a module suite."""
import jax
import jax.numpy as jnp
import jax.scipy.stats as jss

from workshop.jacobian_check import check_change_of_variables, integral_1d, log_abs_det_jacobian


def _gamma_log_pdf(lam):
    return jss.gamma.logpdf(lam, a=2.0, scale=1 / 0.5)


def test_correct_log_transform_gives_zero():
    log_p_phi = lambda phi: _gamma_log_pdf(jnp.exp(phi)) + phi
    d = check_change_of_variables(_gamma_log_pdf, jnp.exp, log_p_phi, jnp.linspace(-2, 3, 11))
    assert jnp.allclose(d, 0.0, atol=1e-10)


def test_missing_jacobian_is_detected_exactly():
    wrong = lambda phi: _gamma_log_pdf(jnp.exp(phi))
    phis = jnp.linspace(-2, 3, 11)
    d = check_change_of_variables(_gamma_log_pdf, jnp.exp, wrong, phis)
    assert jnp.allclose(d, -phis, atol=1e-10)


def test_integral_detects_missing_jacobian():
    grid = jnp.linspace(-10, 6, 40001)
    right = lambda phi: _gamma_log_pdf(jnp.exp(phi)) + phi
    wrong = lambda phi: _gamma_log_pdf(jnp.exp(phi))
    assert jnp.isclose(integral_1d(right, grid), 1.0, atol=1e-6)
    assert not jnp.isclose(integral_1d(wrong, grid), 1.0, atol=1e-2)


def test_multivariate_affine():
    A = jnp.array([[2.0, 0.5], [0.0, 3.0]])
    forward = lambda z: A @ z
    log_p_x = lambda x: jss.multivariate_normal.logpdf(x, jnp.zeros(2), jnp.eye(2))
    log_p_z = lambda z: log_p_x(A @ z) + jnp.log(6.0)
    zs = jax.random.normal(jax.random.key(0), (5, 2))
    d = check_change_of_variables(log_p_x, forward, log_p_z, zs)
    assert jnp.allclose(d, 0.0, atol=1e-10)
    assert jnp.isclose(log_abs_det_jacobian(forward, zs[0]), jnp.log(6.0))
