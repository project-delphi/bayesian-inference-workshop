import jax
import jax.numpy as jnp
import numpy as np

from workshop.m15b_diffusion import dsm_loss, dsm_loss_fn, gaussian_mixture_score, init_score_net, score_net
from tests._util import key


def test_step3_exact_score_beats_zero_score_on_gaussian_data():
    S = jnp.array([[[0.6, 0.2], [0.2, 0.4]]])
    m = jnp.array([[0.3, -0.2]])
    w = jnp.array([1.0])
    x0 = jax.random.multivariate_normal(key(1510), m[0], S[0], (4000,))
    exact = lambda x, t: gaussian_mixture_score(x, t, w, m, S)
    zero = lambda x, t: jnp.zeros_like(x)
    l_exact = dsm_loss_fn(exact, key(1511), x0)
    l_zero = dsm_loss_fn(zero, key(1511), x0)
    # zero score gives E||eps||^2 = d = 2; the exact score is strictly better.
    np.testing.assert_allclose(l_zero, 2.0, atol=0.1)
    assert float(l_exact) < 0.9 * float(l_zero)


def test_step3_network_shapes_and_finite_gradient():
    params = init_score_net(key(1512), dim=2, hidden=32)
    x = jnp.array([0.3, -0.7])
    s = score_net(params, x, jnp.asarray(0.5))
    assert s.shape == (2,)
    x0 = jax.random.normal(key(1513), (64, 2))
    loss, grads = jax.value_and_grad(dsm_loss)(params, key(1514), x0)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))
    # output scaling: near t = 0 the network output is amplified by 1/std(t)
    s_small = score_net(params, x, jnp.asarray(1e-3))
    s_mid = score_net(params, x, jnp.asarray(0.5))
    assert float(jnp.linalg.norm(s_small)) > 5 * float(jnp.linalg.norm(s_mid))
