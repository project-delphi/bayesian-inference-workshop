import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from workshop.m01_jax import ar1_simulate, cumulative_logsumexp
from tests._util import key, mc_close


def test_step5_ar1_matches_python_loop():
    phi, sigma, n = 0.9, 0.3, 50
    xs = ar1_simulate(key(20), phi, sigma, n)
    eps = jax.random.normal(key(20), (n,))
    ref = []
    x = 0.0
    for e in np.asarray(eps):
        x = phi * x + sigma * e
        ref.append(x)
    np.testing.assert_allclose(xs, np.array(ref), rtol=1e-12)
    assert xs.shape == (n,)


def test_step5_ar1_stationary_variance():
    phi, sigma, n = 0.8, 1.0, 200_000
    xs = ar1_simulate(key(21), phi, sigma, n)
    tail = xs[1000:]
    target = sigma**2 / (1 - phi**2)
    # Rough SE for the variance of an AR(1): inflate the iid SE by the IACT.
    iact = (1 + phi**2) / (1 - phi**2)
    se = np.sqrt(2 * target**2 / tail.shape[0] * iact)
    mc_close(tail.var(), target, se, n_se=5, msg="AR(1) variance")


def test_step5_cumulative_logsumexp():
    a = jax.random.normal(key(22), (120,)) * 50 + 1e4
    out = cumulative_logsumexp(a)
    ref = np.array([scipy.special.logsumexp(np.asarray(a[: t + 1])) for t in range(a.shape[0])])
    assert out.shape == a.shape
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(out, ref, rtol=1e-12)
