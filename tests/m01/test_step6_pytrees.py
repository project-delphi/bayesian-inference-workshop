import jax
import jax.numpy as jnp
import numpy as np

from workshop.m01_jax import tree_l2_norm, tree_sgd_step


def _params():
    return {"layer1": {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}, "layer2": [jnp.full((3,), 2.0), (jnp.array(1.0),)]}


def test_step6_structure_preserved_and_values_right():
    p = _params()
    g = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.5, p)
    new = tree_sgd_step(p, g, 0.1)
    assert jax.tree_util.tree_structure(new) == jax.tree_util.tree_structure(p)
    np.testing.assert_allclose(new["layer1"]["w"], 1.0 - 0.05)
    np.testing.assert_allclose(new["layer2"][1][0], 1.0 - 0.05)


def test_step6_descends_a_loss():
    p = _params()
    loss = lambda t: tree_l2_norm(t) ** 2
    for _ in range(50):
        p = tree_sgd_step(p, jax.grad(loss)(p), 0.1)
    assert loss(p) < 1e-3


def test_step6_norm():
    p = _params()
    expected = np.sqrt(6 * 1.0 + 3 * 4.0 + 1.0)
    np.testing.assert_allclose(tree_l2_norm(p), expected, rtol=1e-12)
