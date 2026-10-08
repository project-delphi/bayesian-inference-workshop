import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import single_cell_embedding
from workshop.m03_expfam import Gaussian
from workshop.m06_cavi import GMMPrior, VarParams, elbo, update_dirichlet, update_gaussian_means
from tests._util import key


def test_step3_dirichlet_is_prior_plus_expected_counts():
    r = jnp.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]])
    np.testing.assert_allclose(update_dirichlet(r, 0.5), [0.5 + 1.75, 0.5 + 1.25])


def test_step3_gaussian_update_equals_natural_parameter_addition():
    # Hard assignments: the q(mu_k) update must equal the exact conjugate posterior of
    # N(0, sigma0^2 I) under N_k observations, expressed via Module 3 natural parameters.
    x, z, _, _ = single_cell_embedding()
    x = jnp.asarray(x)
    r = jax.nn.one_hot(jnp.asarray(z), 4)
    sigma0, sigma = 3.0, 0.35
    m, s2 = update_gaussian_means(x, r, sigma0, sigma)
    fam = Gaussian(2)
    eta_prior = fam.to_natural(jnp.zeros(2), sigma0**2 * jnp.eye(2))
    for k in range(4):
        xs = x[jnp.asarray(z) == k]
        # likelihood contribution to the natural parameters of mu: sum_i (x_i/sigma^2, -I/(2 sigma^2))
        eta_lik = jnp.concatenate([xs.sum(0) / sigma**2, (-0.5 * len(xs) * jnp.eye(2) / sigma**2).ravel()])
        mu_post, cov_post = fam.to_standard(eta_prior + eta_lik)
        np.testing.assert_allclose(m[k], mu_post, rtol=1e-10)
        np.testing.assert_allclose(cov_post, s2[k] * jnp.eye(2), rtol=1e-10, atol=1e-12)


def test_step3_updates_increase_elbo():
    x, _, means, _ = single_cell_embedding()
    x = jnp.asarray(x)
    prior = GMMPrior(1.0, 3.0, 0.35)
    r = jax.random.dirichlet(key(220), jnp.ones(4), (600,))
    q = VarParams(r, jnp.ones(4) * 10.0, jnp.zeros((4, 2)), jnp.ones(4))
    e0 = elbo(x, q, prior)
    q1 = q._replace(alpha=update_dirichlet(r, prior.alpha0))
    e1 = elbo(x, q1, prior)
    m, s2 = update_gaussian_means(x, r, prior.sigma0, prior.sigma)
    q2 = q1._replace(m=m, s2=s2)
    e2 = elbo(x, q2, prior)
    assert e1 > e0 and e2 > e1
