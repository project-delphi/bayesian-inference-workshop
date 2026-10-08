import jax.numpy as jnp
import numpy as np

from workshop.m03_expfam import Bernoulli, Categorical, Dirichlet, Gamma
from workshop.m04_kl_bregman import kl, kl_bernoulli, kl_categorical, kl_dirichlet, kl_gamma


def test_step2_bernoulli():
    fam = Bernoulli()
    p, q = jnp.asarray(0.062), jnp.asarray(0.079)
    np.testing.assert_allclose(kl_bernoulli(p, q), kl(fam, fam.to_natural(p), fam.to_natural(q)), rtol=1e-10)


def test_step2_categorical():
    fam = Categorical(6)
    p = jnp.array([0.31, 0.22, 0.14, 0.12, 0.09, 0.12])
    q = jnp.ones(6) / 6
    np.testing.assert_allclose(kl_categorical(p, q), kl(fam, fam.to_natural(p), fam.to_natural(q)), rtol=1e-10)


def test_step2_gamma():
    fam = Gamma()
    a0, b0, a1, b1 = 3.0, 2.0, 1.5, 0.7
    ref = kl(fam, fam.to_natural(jnp.asarray(a0), jnp.asarray(b0)), fam.to_natural(jnp.asarray(a1), jnp.asarray(b1)))
    np.testing.assert_allclose(kl_gamma(a0, b0, a1, b1), ref, rtol=1e-10)


def test_step2_dirichlet():
    fam = Dirichlet(4)
    a0, a1 = jnp.array([2.0, 3.0, 1.5, 0.8]), jnp.array([1.0, 1.0, 4.0, 2.0])
    np.testing.assert_allclose(kl_dirichlet(a0, a1), kl(fam, fam.to_natural(a0), fam.to_natural(a1)), rtol=1e-10)
