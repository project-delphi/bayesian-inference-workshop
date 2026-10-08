import jax
import jax.numpy as jnp
import numpy as np

from workshop.m01_jax import newton_minimize, newton_step


def quad(x):
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, -1.0])
    return 0.5 * x @ A @ x - b @ x


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def test_step3_one_step_solves_quadratic():
    x0 = jnp.array([5.0, -7.0])
    x1 = newton_step(quad, x0)
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, -1.0])
    np.testing.assert_allclose(x1, jnp.linalg.solve(A, b), rtol=1e-10)


def test_step3_scan_trajectory_shape_and_convergence():
    x0 = jnp.array([-1.2, 1.0])
    f = jax.jit(newton_minimize, static_argnums=(0, 2))
    x_final, traj = f(rosenbrock, x0, 12)
    assert traj.shape == (12, 2)
    np.testing.assert_allclose(traj[-1], x_final)
    np.testing.assert_allclose(x_final, jnp.ones(2), atol=1e-8)
    # Newton on Rosenbrock from this start converges; the last iterates should be fixed.
    np.testing.assert_allclose(traj[-1], traj[-2], atol=1e-8)
