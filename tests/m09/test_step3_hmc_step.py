import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import GAUSS2D_COV, gaussian_2d_log_prob, hamiltonian, hmc_step, kinetic_energy, standard_normal_log_prob
from tests._util import key


def test_step3_hamiltonian():
    q = jnp.array([0.5, -0.5])
    p = jnp.array([1.0, 2.0])
    inv_mass = jnp.array([1.0, 0.25])
    np.testing.assert_allclose(kinetic_energy(p, inv_mass), 0.5 * (1.0 + 0.25 * 4.0))
    np.testing.assert_allclose(hamiltonian(q, p, gaussian_2d_log_prob, inv_mass), -gaussian_2d_log_prob(q) + 1.0)


def test_step3_single_transitions_preserve_the_target():
    # Start from exact draws; one HMC transition must leave the distribution invariant.
    n = 20_000
    q0s = jax.random.multivariate_normal(key(210), jnp.zeros(2), GAUSS2D_COV, (n,))
    keys = jax.random.split(key(211), n)
    step = jax.vmap(lambda k, q: hmc_step(gaussian_2d_log_prob, k, q, 0.4, 5, jnp.ones(2)))
    q1, info = jax.jit(step)(keys, q0s)
    assert info.accept.dtype == jnp.bool_
    assert 0.7 < float(info.accept.mean()) <= 1.0
    np.testing.assert_allclose(q1.mean(0), 0.0, atol=0.04)
    np.testing.assert_allclose(jnp.cov(q1.T), GAUSS2D_COV, atol=0.08)
    np.testing.assert_allclose(info.log_prob, jax.vmap(gaussian_2d_log_prob)(q1), rtol=1e-8)


def test_step3_rejects_nonfinite_energy():
    bad = lambda q: jnp.where(jnp.abs(q[0]) > 1.0, -jnp.inf, standard_normal_log_prob(q))
    q, info = hmc_step(bad, key(212), jnp.array([0.9]), 0.5, 10, jnp.ones(1))
    assert jnp.isfinite(q).all()
    if not bool(info.accept):
        assert float(info.accept_prob) == 0.0 or jnp.isinf(info.energy_error)
