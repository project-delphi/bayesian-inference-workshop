import jax.numpy as jnp
import numpy as np

from workshop.m03_expfam import Categorical, Dirichlet, Gamma, Gaussian
from workshop.m04_kl_bregman import kl, kl_monte_carlo
from tests._util import key, mc_close

N = 200_000


def test_step1_kl_zero_and_nonnegative():
    fam = Gamma()
    e1 = fam.to_natural(jnp.asarray(3.0), jnp.asarray(2.0))
    e2 = fam.to_natural(jnp.asarray(1.5), jnp.asarray(0.7))
    np.testing.assert_allclose(kl(fam, e1, e1), 0.0, atol=1e-12)
    assert kl(fam, e1, e2) > 0 and kl(fam, e2, e1) > 0
    assert not np.isclose(kl(fam, e1, e2), kl(fam, e2, e1))


def test_step1_bregman_matches_mc_gamma():
    fam = Gamma()
    e1 = fam.to_natural(jnp.asarray(3.0), jnp.asarray(2.0))
    e2 = fam.to_natural(jnp.asarray(1.5), jnp.asarray(0.7))
    est, se = kl_monte_carlo(fam, e1, e2, key(100), N)
    mc_close(kl(fam, e1, e2), est, se, msg="gamma KL")


def test_step1_bregman_matches_mc_dirichlet_and_categorical():
    fam = Dirichlet(3)
    e1, e2 = fam.to_natural(jnp.array([2.0, 3.0, 1.5])), fam.to_natural(jnp.array([1.0, 1.0, 4.0]))
    est, se = kl_monte_carlo(fam, e1, e2, key(101), N)
    mc_close(kl(fam, e1, e2), est, se, msg="dirichlet KL")
    cat = Categorical(4)
    c1, c2 = cat.to_natural(jnp.array([0.4, 0.3, 0.2, 0.1])), cat.to_natural(jnp.array([0.1, 0.2, 0.3, 0.4]))
    est, se = kl_monte_carlo(cat, c1, c2, key(102), N)
    mc_close(kl(cat, c1, c2), est, se, msg="categorical KL")


def test_step1_gaussian_matches_module2_closed_form():
    from workshop.m02_diagnostic import kl_gaussians

    fam = Gaussian(2)
    mu0, cov0 = jnp.zeros(2), jnp.array([[1.0, 0.3], [0.3, 0.5]])
    mu1, cov1 = jnp.array([0.5, -0.2]), jnp.array([[2.0, -0.4], [-0.4, 1.0]])
    e0, e1 = fam.to_natural(mu0, cov0), fam.to_natural(mu1, cov1)
    np.testing.assert_allclose(kl(fam, e0, e1), kl_gaussians(mu0, cov0, mu1, cov1), rtol=1e-8)
