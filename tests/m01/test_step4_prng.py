import jax
import jax.numpy as jnp
import numpy as np

from workshop.m01_jax import sample_gaussian_mixture
from tests._util import key, mc_close

W = jnp.array([0.2, 0.5, 0.3])
MU = jnp.array([-3.0, 0.0, 4.0])
S = jnp.array([0.5, 1.0, 0.7])


def test_step4_deterministic_in_key():
    x1, z1 = sample_gaussian_mixture(key(10), W, MU, S, 1000)
    x2, z2 = sample_gaussian_mixture(key(10), W, MU, S, 1000)
    x3, _ = sample_gaussian_mixture(key(11), W, MU, S, 1000)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(z1, z2)
    assert not np.allclose(x1, x3)


def test_step4_moments():
    n = 200_000
    x, z = sample_gaussian_mixture(key(12), W, MU, S, n)
    assert x.shape == (n,) and z.shape == (n,)
    mean = W @ MU
    var = W @ (S**2 + MU**2) - mean**2
    mc_close(x.mean(), mean, np.sqrt(var / n), msg="mixture mean")
    freqs = jnp.bincount(z, length=3) / n
    mc_close(freqs, W, np.sqrt(W * (1 - W) / n), msg="component frequencies")


def test_step4_components_are_conditionally_gaussian():
    x, z = sample_gaussian_mixture(key(13), W, MU, S, 100_000)
    for k in range(3):
        xk = x[z == k]
        mc_close(xk.mean(), MU[k], S[k] / np.sqrt(xk.shape[0]), msg=f"component {k} mean")
