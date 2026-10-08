import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m07_gradients import churn_log_joint, laplace, laplace_log_evidence
from tests.m07._common import X, beta_true, log_joint, y


def test_step1_log_joint_matches_scipy():
    b = jnp.array([0.1, -0.2, 0.3, 0.0, 0.5])
    p = 1 / (1 + np.exp(-np.asarray(X @ b)))
    ref = scipy.stats.bernoulli(p).logpmf(np.asarray(y)).sum() + scipy.stats.norm(0, 2.5).logpdf(np.asarray(b)).sum()
    np.testing.assert_allclose(churn_log_joint(b, X, y), ref, rtol=1e-10)


def test_step1_laplace_is_a_stationary_point_with_correct_curvature():
    mean, cov = laplace(log_joint, jnp.zeros(5))
    np.testing.assert_allclose(jax.grad(log_joint)(mean), 0.0, atol=1e-8)
    np.testing.assert_allclose(cov, jnp.linalg.inv(-jax.hessian(log_joint)(mean)), rtol=1e-10)
    sd = jnp.sqrt(jnp.diag(cov))
    assert bool(jnp.all(jnp.abs(mean - beta_true) < 3 * sd))


def test_step1_evidence_exact_for_gaussian():
    m = jnp.array([0.3, -1.0])
    S = jnp.array([[1.0, 0.4], [0.4, 2.0]])
    logC = -3.7
    lj = lambda z: jax.scipy.stats.multivariate_normal.logpdf(z, m, S) + logC
    mean, cov = laplace(lj, jnp.zeros(2))
    np.testing.assert_allclose(mean, m, atol=1e-8)
    np.testing.assert_allclose(laplace_log_evidence(lj, mean, cov), logC, rtol=1e-8)
