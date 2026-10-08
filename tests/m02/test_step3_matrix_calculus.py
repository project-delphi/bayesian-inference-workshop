import jax
import jax.numpy as jnp
import numpy as np

from workshop.m02_diagnostic import grad_logdet, grad_quadratic_form_x, grad_trace_inv
from tests._util import key


def _spd(k, d):
    A = jax.random.normal(k, (d, d))
    return A @ A.T + d * jnp.eye(d)


def test_step3_logdet():
    A = _spd(key(40), 4)
    np.testing.assert_allclose(grad_logdet(A), jax.grad(lambda M: jnp.linalg.slogdet(M)[1])(A), rtol=1e-9)
    B = jax.random.normal(key(41), (4, 4)) + 4 * jnp.eye(4)  # non-symmetric
    np.testing.assert_allclose(grad_logdet(B), jax.grad(lambda M: jnp.linalg.slogdet(M)[1])(B), rtol=1e-9)


def test_step3_quadratic_form():
    A = jax.random.normal(key(42), (5, 5))
    x = jax.random.normal(key(43), (5,))
    np.testing.assert_allclose(grad_quadratic_form_x(A, x), jax.grad(lambda v: v @ A @ v)(x), rtol=1e-10)


def test_step3_trace_inverse():
    A = _spd(key(44), 3)
    B = jax.random.normal(key(45), (3, 3))
    np.testing.assert_allclose(grad_trace_inv(A, B), jax.grad(lambda M: jnp.trace(jnp.linalg.solve(M, B)))(A), rtol=1e-9)
