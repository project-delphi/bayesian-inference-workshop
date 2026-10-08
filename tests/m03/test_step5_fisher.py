import jax
import jax.numpy as jnp
import numpy as np

from workshop.m03_expfam import Bernoulli, Categorical, Dirichlet, Gamma, Gaussian
from tests._util import key, mc_close

N = 200_000


def _mc_cov_stats(fam, eta, k):
    xs = fam.sample(k, eta, N)
    T = jax.vmap(fam.sufficient_stats)(xs)
    Tc = T - T.mean(0)
    cov = Tc.T @ Tc / (N - 1)
    # crude SE for covariance entries
    se = np.sqrt(((Tc[:, :, None] * Tc[:, None, :]) ** 2).mean(0) / N)
    return cov, se


def test_step5_bernoulli_fisher_is_variance():
    fam = Bernoulli()
    p = 0.3
    eta = fam.to_natural(jnp.asarray(p))
    np.testing.assert_allclose(fam.fisher(eta), [[p * (1 - p)]], rtol=1e-10)


def test_step5_categorical_fisher_matches_mc_and_is_pd():
    fam = Categorical(4)
    eta = fam.to_natural(jnp.array([0.4, 0.3, 0.2, 0.1]))
    F = fam.fisher(eta)
    assert F.shape == (3, 3)
    assert bool(jnp.all(jnp.linalg.eigvalsh(F) > 0))
    cov, se = _mc_cov_stats(fam, eta, key(90))
    mc_close(F, cov, se, msg="categorical fisher")


def test_step5_gamma_fisher_matches_mc():
    fam = Gamma()
    eta = fam.to_natural(jnp.asarray(3.5), jnp.asarray(1.7))
    cov, se = _mc_cov_stats(fam, eta, key(91))
    mc_close(fam.fisher(eta), cov, se, msg="gamma fisher")


def test_step5_dirichlet_fisher_matches_mc():
    fam = Dirichlet(3)
    eta = fam.to_natural(jnp.array([2.0, 3.0, 1.5]))
    cov, se = _mc_cov_stats(fam, eta, key(92))
    mc_close(fam.fisher(eta), cov, se, msg="dirichlet fisher")


def test_step5_gaussian_fisher_mean_block_is_covariance():
    d = 2
    cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    fam = Gaussian(d)
    eta = fam.to_natural(jnp.zeros(d), cov)
    F = fam.fisher(eta)
    np.testing.assert_allclose(F[:d, :d], cov, rtol=1e-8)


def test_step5_score_has_zero_mean_and_outer_product_is_fisher():
    fam = Gamma()
    eta = fam.to_natural(jnp.asarray(3.5), jnp.asarray(1.7))
    xs = fam.sample(key(93), eta, N)
    S = jax.vmap(fam.score, in_axes=(0, None))(xs, eta)
    mc_close(S.mean(0), jnp.zeros(2), S.std(0) / np.sqrt(N), msg="E[score]")
    outer = jnp.einsum("ni,nj->ij", S, S) / N
    se = np.sqrt(((S[:, :, None] * S[:, None, :]) ** 2).mean(0) / N)
    mc_close(outer, fam.fisher(eta), se, msg="E[score score^T]")
