import jax
import jax.numpy as jnp
import numpy as np

from workshop.m15a_flows import flow_forward, flow_inverse, flow_log_prob, flow_sample, init_flow
from tests._util import key


def _perturbed(d, k):
    p = init_flow(key(20), d, 4, 16)
    return jax.tree_util.tree_map(lambda a: a + 0.3 * jax.random.normal(k, a.shape), p)


def test_step2_identity_at_init():
    p = init_flow(key(21), 10, 4)
    z = jax.random.normal(key(22), (10,))
    x, ld = flow_forward(p, z)
    np.testing.assert_allclose(x, z)
    assert float(ld) == 0.0
    np.testing.assert_allclose(flow_log_prob(p, z), -0.5 * z @ z - 5 * jnp.log(2 * jnp.pi), rtol=1e-12)


def test_step2_inverse_and_logdet_full_stack():
    p = _perturbed(10, key(23))
    z = jax.random.normal(key(24), (10,))
    x, ld = flow_forward(p, z)
    z2, ld_inv = flow_inverse(p, x)
    np.testing.assert_allclose(z2, z, atol=1e-8)
    np.testing.assert_allclose(ld + ld_inv, 0.0, atol=1e-10)
    J = jax.jacfwd(lambda v: flow_forward(p, v)[0])(z)
    np.testing.assert_allclose(ld, jnp.linalg.slogdet(J)[1], atol=1e-8)
    assert not np.allclose(x, z)


def test_step2_log_prob_integrates_to_one_in_2d():
    p = init_flow(key(20), 2, 4, 16)
    p = jax.tree_util.tree_map(lambda a: a + 0.15 * jax.random.normal(key(25), a.shape), p)
    g = jnp.linspace(-12, 12, 601)
    X, Y = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], 1)
    dens = jnp.exp(jax.vmap(lambda v: flow_log_prob(p, v))(pts)).reshape(X.shape)
    h = float(g[1] - g[0])
    total = float(jnp.trapezoid(jnp.trapezoid(dens, dx=h, axis=1), dx=h))
    np.testing.assert_allclose(total, 1.0, atol=2e-3)


def test_step2_sample_log_q_matches_log_prob():
    p = _perturbed(10, key(26))
    x, log_q = flow_sample(key(27), p, 50)
    assert x.shape == (50, 10) and log_q.shape == (50,)
    np.testing.assert_allclose(log_q, jax.vmap(lambda v: flow_log_prob(p, v))(x), atol=1e-8)
