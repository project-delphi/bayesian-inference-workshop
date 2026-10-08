import jax
import jax.numpy as jnp
import numpy as np

from workshop.m08_bbvi import adam_init, adam_update


def test_step2_single_step_matches_formula():
    params = {"a": jnp.array([1.0, -2.0]), "b": jnp.array(0.5)}
    grads = {"a": jnp.array([0.3, -0.1]), "b": jnp.array(2.0)}
    state = adam_init(params)
    new, state = adam_update(grads, state, params, lr=0.1)
    # first step: m_hat = g, v_hat = g^2, so step = lr * sign(g) (up to eps)
    np.testing.assert_allclose(new["a"], params["a"] + 0.1 * jnp.sign(grads["a"]), rtol=1e-6)
    np.testing.assert_allclose(new["b"], 0.5 + 0.1, rtol=1e-6)
    assert int(state["t"]) == 1


def test_step2_ascends_a_concave_objective():
    f = lambda p: -jnp.sum((p["x"] - 3.0) ** 2) - (p["y"] + 1.0) ** 2
    params = {"x": jnp.zeros(3), "y": jnp.array(0.0)}
    state = adam_init(params)

    def body(carry, _):
        p, s = carry
        g = jax.grad(f)(p)
        p, s = adam_update(g, s, p, lr=0.05)
        return (p, s), f(p)

    (params, _), trace = jax.lax.scan(body, (params, state), None, length=600)
    np.testing.assert_allclose(params["x"], 3.0, atol=1e-2)
    np.testing.assert_allclose(params["y"], -1.0, atol=1e-2)
    assert trace[-1] > trace[0]
