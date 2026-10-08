import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import standard_normal_log_prob
from workshop.m10_adaptation import trajectory_length_study, uturn


def test_step5_uturn_criterion():
    inv_mass = jnp.ones(2)
    q_minus, q_plus = jnp.zeros(2), jnp.array([1.0, 0.0])
    assert not bool(uturn(q_minus, q_plus, jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]), inv_mass))
    assert bool(uturn(q_minus, q_plus, jnp.array([1.0, 0.0]), jnp.array([-1.0, 0.0]), inv_mass))
    assert bool(uturn(q_minus, q_plus, jnp.array([-1.0, 0.0]), jnp.array([1.0, 0.0]), inv_mass))


def test_step5_uturn_fires_at_half_period_on_gaussian():
    # Harmonic oscillator: period 2 pi; the endpoints start moving toward each other after
    # half a period, pi / eps steps.
    eps = 0.05
    flags, first = trajectory_length_study(standard_normal_log_prob, jnp.array([1.0]), jnp.array([0.0]), eps, 200, jnp.ones(1))
    assert flags.shape == (200,)
    expected = np.pi / eps
    assert abs(float(first) - expected) <= 2
    # anisotropic target: u-turn in the slow direction takes about 10x longer
    var = jnp.array([1.0, 100.0])
    lp = lambda z: -0.5 * jnp.sum(z**2 / var)
    _, first_fast = trajectory_length_study(lp, jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), eps, 1000, jnp.ones(2))
    _, first_slow = trajectory_length_study(lp, jnp.array([0.0, 10.0]), jnp.array([0.0, 0.0]), eps, 1000, jnp.ones(2))
    assert float(first_slow) > 5 * float(first_fast)
