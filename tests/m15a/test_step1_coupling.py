import jax
import jax.numpy as jnp
import numpy as np

from workshop.m15a_flows import coupling_forward, coupling_inverse, init_mlp, make_masks, mlp
from tests._util import key

D = 6


def _perturbed(k):
    p = init_mlp(k, [D, 16, 16, 2 * D])
    return jax.tree_util.tree_map(lambda a: a + 0.4 * jax.random.normal(key(1), a.shape), p)


def test_step1_mlp_init_shapes_and_zero_last_layer():
    p = init_mlp(key(10), [D, 16, 16, 2 * D])
    assert [W.shape for W, _ in p] == [(D, 16), (16, 16), (16, 2 * D)]
    assert bool(jnp.all(p[-1][0] == 0)) and bool(jnp.all(p[-1][1] == 0))
    assert not bool(jnp.all(p[0][0] == 0))
    x = jax.random.normal(key(11), (D,))
    np.testing.assert_array_equal(mlp(p, x), jnp.zeros(2 * D))


def test_step1_identity_at_init():
    p = init_mlp(key(12), [D, 16, 16, 2 * D])
    mask = make_masks(D, 1)[0]
    z = jax.random.normal(key(13), (D,))
    x, ld = coupling_forward(p, mask, z)
    np.testing.assert_allclose(x, z)
    assert float(ld) == 0.0


def test_step1_inverse_and_logdet():
    p = _perturbed(key(14))
    mask = make_masks(D, 2)[1]
    z = jax.random.normal(key(15), (D,)) * 2
    x, ld = coupling_forward(p, mask, z)
    z2, ld_inv = coupling_inverse(p, mask, x)
    np.testing.assert_allclose(z2, z, atol=1e-8)
    np.testing.assert_allclose(ld + ld_inv, 0.0, atol=1e-10)
    J = jax.jacfwd(lambda v: coupling_forward(p, mask, v)[0])(z)
    np.testing.assert_allclose(ld, jnp.linalg.slogdet(J)[1], atol=1e-8)
    # conditioned coordinates are untouched
    np.testing.assert_array_equal(x[mask == 1], z[mask == 1])
    assert not np.allclose(x[mask == 0], z[mask == 0])


def test_step1_masks_alternate_and_cover():
    m = make_masks(D, 4)
    assert m.shape == (4, D)
    np.testing.assert_array_equal(m[0], jnp.array([1, 0, 1, 0, 1, 0.0]))
    np.testing.assert_array_equal(m[1], 1 - m[0])
    np.testing.assert_array_equal(m[2], m[0])
    assert bool(jnp.all(m.sum(0) == 2))
