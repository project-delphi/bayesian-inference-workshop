import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.data import single_cell_embedding
from workshop.m06_cavi import GMMPrior, VarParams, categorical_entropy, elbo, expected_log_lik, expected_log_prior_z, kl_gaussian_means
from tests._util import key, mc_close


def _setup():
    x, _, means, _ = single_cell_embedding()
    x = jnp.asarray(x)
    K, d = 4, 2
    r = jax.random.dirichlet(key(200), jnp.ones(K), (x.shape[0],))
    alpha = jnp.array([120.0, 80.0, 60.0, 40.0])
    m = jnp.asarray(means) + 0.1
    s2 = jnp.array([0.01, 0.02, 0.03, 0.04])
    return x, VarParams(r, alpha, m, s2), GMMPrior(1.0, 3.0, 0.35)


def test_step1_expected_log_lik_matches_mc():
    x, q, prior = _setup()
    N = 20_000
    mus = q.m[None] + jnp.sqrt(q.s2)[None, :, None] * jax.random.normal(key(201), (N, 4, 2))  # [N, K, d]
    # E over mu of sum_i sum_k r_ik log N(x_i | mu_k, sigma^2 I)
    def one(mu):
        ll = scipy_ll(x, mu, prior.sigma)
        return jnp.sum(q.r * ll)

    vals = jax.vmap(one)(mus)
    mc_close(expected_log_lik(x, q.r, q.m, q.s2, prior.sigma), vals.mean(), vals.std() / np.sqrt(N), msg="E[log lik]")


def scipy_ll(x, mu, sigma):
    d = x.shape[1]
    sq = jnp.sum((x[:, None, :] - mu[None, :, :]) ** 2, axis=-1)
    return -0.5 * d * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * sq / sigma**2


def test_step1_expected_log_prior_z_matches_mc():
    x, q, prior = _setup()
    N = 200_000
    pis = jax.random.dirichlet(key(202), q.alpha, (N,))
    vals = jnp.log(pis) @ q.r.sum(0)
    mc_close(expected_log_prior_z(q.r, q.alpha), vals.mean(), vals.std() / np.sqrt(N), msg="E[log p(z|pi)]")


def test_step1_entropy_and_kl_means():
    r = jnp.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    ref = sum(scipy.stats.entropy(row) for row in np.asarray(r))
    np.testing.assert_allclose(categorical_entropy(r), ref, rtol=1e-12)
    m = jnp.array([[1.0, -2.0], [0.5, 0.5]])
    s2 = jnp.array([0.1, 0.4])
    ref = 0.0
    for k in range(2):
        ref += np.sum([scipy_kl_1d(m[k, j], s2[k], 3.0) for j in range(2)])
    np.testing.assert_allclose(kl_gaussian_means(m, s2, 3.0), ref, rtol=1e-10)


def scipy_kl_1d(mu, var, sigma0):
    return 0.5 * (var / sigma0**2 + mu**2 / sigma0**2 - 1 + np.log(sigma0**2 / var))


def test_step1_elbo_is_a_lower_bound_on_a_tiny_problem():
    # 3 points, K=2, d=1: evidence by quadrature over mu (pi integrated analytically via
    # the Dirichlet-multinomial for each z configuration).
    x = jnp.array([[-1.0], [0.0], [1.2]])
    prior = GMMPrior(1.0, 2.0, 0.5)
    r = jnp.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
    alpha = jnp.array([2.4, 2.6])
    m = jnp.array([[-0.5], [0.6]])
    s2 = jnp.array([0.2, 0.2])
    val = elbo(x, VarParams(r, alpha, m, s2), prior)
    # exact log evidence
    import itertools
    from scipy.special import gammaln

    total = -np.inf
    for z in itertools.product(range(2), repeat=3):
        counts = np.bincount(z, minlength=2)
        log_pz = gammaln(2.0) - gammaln(2.0 + 3) + np.sum(gammaln(1.0 + counts) - gammaln(1.0))
        log_px = 0.0
        for k in range(2):
            xs_k = np.asarray(x)[np.array(z) == k, 0]
            n_k = len(xs_k)
            if n_k == 0:
                continue
            # marginal of n_k points under mu ~ N(0, sigma0^2), x | mu ~ N(mu, sigma^2)
            cov = prior.sigma**2 * np.eye(n_k) + prior.sigma0**2 * np.ones((n_k, n_k))
            log_px += scipy.stats.multivariate_normal(np.zeros(n_k), cov).logpdf(xs_k)
        total = np.logaddexp(total, log_pz + log_px)
    assert float(val) < total
    assert total - float(val) < 5.0
