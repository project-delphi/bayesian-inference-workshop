import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import gaussian_2d_log_prob, leapfrog
from tests._util import key

grad = jax.grad(gaussian_2d_log_prob)
INV_MASS = jnp.array([1.0, 0.5])


def test_step2_reversibility():
    q0 = jnp.array([0.3, -1.2])
    p0 = jnp.array([0.7, 0.4])
    q1, p1 = leapfrog(grad, q0, p0, 0.1, 25, INV_MASS)
    q2, p2 = leapfrog(grad, q1, -p1, 0.1, 25, INV_MASS)
    np.testing.assert_allclose(q2, q0, atol=1e-8)
    np.testing.assert_allclose(-p2, p0, atol=1e-8)


def test_step2_volume_preservation():
    z0 = jnp.array([0.3, -1.2, 0.7, 0.4])

    def one_step(z):
        q, p = leapfrog(grad, z[:2], z[2:], 0.3, 1, INV_MASS)
        return jnp.concatenate([q, p])

    J = jax.jacfwd(one_step)(z0)
    np.testing.assert_allclose(jnp.abs(jnp.linalg.det(J)), 1.0, atol=1e-10)


def test_step2_energy_error_is_second_order():
    q0 = jnp.array([0.3, -1.2])
    p0 = jnp.array([0.7, 0.4])
    H = lambda q, p: -gaussian_2d_log_prob(q) + 0.5 * jnp.sum(INV_MASS * p**2)
    T = 2.0
    errs = []
    for eps in (0.2, 0.1, 0.05):
        q, p = leapfrog(grad, q0, p0, eps, int(round(T / eps)), INV_MASS)
        errs.append(abs(float(H(q, p) - H(q0, p0))))
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0
    assert errs[0] / errs[1] < 5.5 and errs[1] / errs[2] < 5.5
