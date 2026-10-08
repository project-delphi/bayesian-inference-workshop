import jax
import jax.numpy as jnp
import numpy as np

from workshop.m03_expfam import Bernoulli, Categorical, Dirichlet, Gamma
from workshop.m04_kl_bregman import conjugate, entropy_monte_carlo, natural_from_mean
from tests._util import key, mc_close


def test_step3_inversion_roundtrip_all_families():
    cases = [
        (Bernoulli(), Bernoulli().to_natural(jnp.asarray(0.23)), jnp.zeros(1)),
        (Categorical(4), Categorical(4).to_natural(jnp.array([0.4, 0.3, 0.2, 0.1])), jnp.zeros(3)),
        (Gamma(), Gamma().to_natural(jnp.asarray(3.5), jnp.asarray(1.7)), jnp.array([0.0, -1.0])),
        (Dirichlet(3), Dirichlet(3).to_natural(jnp.array([2.0, 3.0, 1.5])), jnp.zeros(3)),
    ]
    for fam, eta, eta0 in cases:
        mu = fam.mean_params(eta)
        eta_hat = natural_from_mean(fam, mu, eta0)
        np.testing.assert_allclose(eta_hat, eta, rtol=1e-7, atol=1e-8, err_msg=type(fam).__name__)


def test_step3_conjugate_is_negative_entropy():
    fam = Gamma()
    eta = fam.to_natural(jnp.asarray(3.5), jnp.asarray(1.7))
    mu = fam.mean_params(eta)
    H, se = entropy_monte_carlo(fam, eta, key(110), 200_000)
    mc_close(conjugate(fam, mu, jnp.array([0.0, -1.0])), -H, se, msg="A*(mu) = -H")


def test_step3_gradient_of_conjugate_is_natural_parameter():
    fam = Dirichlet(3)
    eta = fam.to_natural(jnp.array([2.0, 3.0, 1.5]))
    mu = fam.mean_params(eta)
    g = jax.grad(lambda m: conjugate(fam, m, jnp.zeros(3)))(mu)
    np.testing.assert_allclose(g, eta, rtol=1e-6, atol=1e-7)
