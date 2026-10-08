import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m03_expfam import Dirichlet, Gamma
from tests._util import key


def test_step3_gamma_matches_scipy():
    fam = Gamma()
    alpha, beta = 3.5, 1.7
    eta = fam.to_natural(jnp.asarray(alpha), jnp.asarray(beta))
    xs = jnp.array([0.1, 0.8, 2.3, 7.9])
    out = jax.vmap(fam.log_prob, in_axes=(0, None))(xs, eta)
    ref = scipy.stats.gamma(a=alpha, scale=1 / beta).logpdf(np.asarray(xs))
    np.testing.assert_allclose(out, ref, rtol=1e-10)
    a2, b2 = fam.to_standard(eta)
    np.testing.assert_allclose([a2, b2], [alpha, beta], rtol=1e-12)


def test_step3_dirichlet_matches_scipy():
    alpha = jnp.array([2.0, 0.7, 5.0, 1.3])
    fam = Dirichlet(4)
    eta = fam.to_natural(alpha)
    xs = jax.random.dirichlet(key(70), alpha, (6,))
    out = jax.vmap(fam.log_prob, in_axes=(0, None))(xs, eta)
    ref = scipy.stats.dirichlet(np.asarray(alpha)).logpdf(np.asarray(xs).T)
    np.testing.assert_allclose(out, ref, rtol=1e-8)
    np.testing.assert_allclose(fam.to_standard(eta), alpha, rtol=1e-12)
