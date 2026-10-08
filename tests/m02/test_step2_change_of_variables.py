import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m02_diagnostic import pushforward_logpdf


def test_step2_lognormal():
    mu, sd = 0.3, 0.7
    log_p_x = lambda x: jnp.sum(-0.5 * ((x - mu) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2 * jnp.pi))
    f_inv = jnp.log  # y = exp(x)
    y = jnp.array([0.4, 1.3, 2.9])
    out = pushforward_logpdf(log_p_x, f_inv, y)
    ref = scipy.stats.lognorm(s=sd, scale=np.exp(mu)).logpdf(np.asarray(y)).sum()
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_step2_linear_map_in_2d():
    A = jnp.array([[2.0, 0.5], [-1.0, 1.5]])
    log_p_x = lambda x: -0.5 * x @ x - jnp.log(2 * jnp.pi)
    f_inv = lambda y: jnp.linalg.solve(A, y)
    y = jnp.array([0.7, -0.2])
    out = pushforward_logpdf(log_p_x, f_inv, y)
    ref = scipy.stats.multivariate_normal(np.zeros(2), np.asarray(A @ A.T)).logpdf(np.asarray(y))
    np.testing.assert_allclose(out, ref, rtol=1e-10)
