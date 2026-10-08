import jax
import jax.numpy as jnp
import numpy as np

from workshop.m10_adaptation import welford_finalize, welford_init, welford_update
from tests._util import key


def test_step2_matches_numpy_variance():
    xs = jax.random.normal(key(240), (300, 3)) * jnp.array([1.0, 3.0, 0.1]) + 2.0
    s = welford_init(3)
    s = jax.lax.fori_loop(0, 300, lambda i, s: welford_update(s, xs[i]), s)
    np.testing.assert_allclose(s.mean, xs.mean(0), rtol=1e-10)
    np.testing.assert_allclose(welford_finalize(s, regularize=False), xs.var(0, ddof=1), rtol=1e-10)


def test_step2_regularisation_shrinks_toward_identity():
    s = welford_init(2)
    for x in jnp.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]):
        s = welford_update(s, x)
    n = 4.0
    raw = welford_finalize(s, regularize=False)
    reg = welford_finalize(s)
    np.testing.assert_allclose(reg, n / (n + 5) * raw + 1e-3 * 5 / (n + 5), rtol=1e-12)
