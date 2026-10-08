import jax
import jax.numpy as jnp
import numpy as np

from workshop.m03_expfam import Categorical, Gamma
from workshop.m04_kl_bregman import kl, kl_quadratic_approx, natural_from_mean, natural_gradient


def test_step4_kl_locally_quadratic():
    fam = Gamma()
    eta = fam.to_natural(jnp.asarray(3.5), jnp.asarray(1.7))
    delta = jnp.array([0.02, -0.01])
    exact = kl(fam, eta, eta + delta)
    approx = kl_quadratic_approx(fam, eta, delta)
    assert approx > 0
    # error is third order in delta
    np.testing.assert_allclose(exact, approx, rtol=2e-2)
    exact2 = kl(fam, eta, eta + delta / 10)
    approx2 = kl_quadratic_approx(fam, eta, delta / 10)
    assert abs(exact2 - approx2) < abs(exact - approx) / 500


def test_step4_natural_gradient_equals_mean_parameter_gradient():
    fam = Categorical(4)
    eta = fam.to_natural(jnp.array([0.4, 0.3, 0.2, 0.1]))
    f = lambda e: jnp.sum(jnp.sin(e) * jnp.arange(1, 4))
    g_eta = jax.grad(f)(eta)
    mu = fam.mean_params(eta)
    g_mu = jax.grad(lambda m: f(natural_from_mean(fam, m, jnp.zeros(3))))(mu)
    np.testing.assert_allclose(natural_gradient(fam, eta, g_eta), g_mu, rtol=1e-6, atol=1e-7)
