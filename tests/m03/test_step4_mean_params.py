import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma

from workshop.m03_expfam import Bernoulli, Categorical, Dirichlet, Gamma, Gaussian
from tests._util import key, mc_close

N = 200_000


def _mc_mean_stats(fam, eta, k):
    xs = fam.sample(k, eta, N)
    T = jax.vmap(fam.sufficient_stats)(xs)
    return T.mean(0), T.std(0) / np.sqrt(N)


def test_step4_bernoulli_analytic():
    fam = Bernoulli()
    eta = fam.to_natural(jnp.asarray(0.062))
    np.testing.assert_allclose(fam.mean_params(eta), [0.062], rtol=1e-10)


def test_step4_categorical_mc():
    fam = Categorical(6)
    eta = fam.to_natural(jnp.array([0.31, 0.22, 0.14, 0.12, 0.09, 0.12]))
    m, se = _mc_mean_stats(fam, eta, key(80))
    mc_close(fam.mean_params(eta), m, se, msg="categorical mean params")


def test_step4_gaussian_analytic_and_mc():
    d = 2
    mu = jnp.array([0.5, -1.0])
    cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    fam = Gaussian(d)
    eta = fam.to_natural(mu, cov)
    mp = fam.mean_params(eta)
    np.testing.assert_allclose(mp[:d], mu, rtol=1e-8)
    np.testing.assert_allclose(mp[d:].reshape(d, d), cov + jnp.outer(mu, mu), rtol=1e-8)
    m, se = _mc_mean_stats(fam, eta, key(81))
    mc_close(mp, m, se, msg="gaussian mean params")


def test_step4_gamma_analytic():
    fam = Gamma()
    alpha, beta = 3.5, 1.7
    eta = fam.to_natural(jnp.asarray(alpha), jnp.asarray(beta))
    np.testing.assert_allclose(fam.mean_params(eta), [digamma(alpha) - jnp.log(beta), alpha / beta], rtol=1e-10)


def test_step4_dirichlet_analytic_and_mc():
    alpha = jnp.array([2.0, 0.7, 5.0, 1.3])
    fam = Dirichlet(4)
    eta = fam.to_natural(alpha)
    np.testing.assert_allclose(fam.mean_params(eta), digamma(alpha) - digamma(alpha.sum()), rtol=1e-10)
    m, se = _mc_mean_stats(fam, eta, key(82))
    mc_close(fam.mean_params(eta), m, se, msg="dirichlet mean params")
