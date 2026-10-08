import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate

from workshop.m15b_diffusion import BETA_MAX, BETA_MIN, alpha_bar, beta, perturbation_kernel, sample_forward
from tests._util import key


def test_step1_alpha_bar_matches_numerical_integral():
    for t in (0.05, 0.3, 0.7, 1.0):
        integral, _ = scipy.integrate.quad(lambda s: BETA_MIN + s * (BETA_MAX - BETA_MIN), 0, t)
        np.testing.assert_allclose(alpha_bar(jnp.asarray(t)), np.exp(-integral), rtol=1e-10)
    np.testing.assert_allclose(beta(jnp.asarray(0.0)), BETA_MIN)
    np.testing.assert_allclose(beta(jnp.asarray(1.0)), BETA_MAX)


def test_step1_variance_preserving():
    # If x0 ~ N(0, I) then x_t ~ N(0, I) for every t.
    for t in (0.1, 0.5, 0.9):
        mean, std = perturbation_kernel(jnp.ones(2), jnp.asarray(t))
        ab = alpha_bar(jnp.asarray(t))
        np.testing.assert_allclose(mean, jnp.sqrt(ab) * jnp.ones(2), rtol=1e-12)
        np.testing.assert_allclose(ab + std**2, 1.0, rtol=1e-12)


def test_step1_terminal_is_standard_normal_and_sampler_matches_kernel():
    mean, std = perturbation_kernel(jnp.array([3.0, -2.0]), jnp.asarray(1.0))
    # sqrt(alpha_bar(1)) = exp(-5.025) ~ 0.0066, so the signal of a 3-unit input is ~0.02
    assert float(jnp.abs(mean).max()) < 0.05 and abs(float(std) - 1.0) < 1e-3
    x0 = jnp.array([1.5, -0.5])
    t = jnp.asarray(0.4)
    xs = jax.vmap(lambda k: sample_forward(k, x0, t))(jax.random.split(key(1500), 100_000))
    m, s = perturbation_kernel(x0, t)
    np.testing.assert_allclose(xs.mean(0), m, atol=4 * float(s) / np.sqrt(100_000) + 1e-9)
    np.testing.assert_allclose(xs.std(0), s, rtol=0.02)
