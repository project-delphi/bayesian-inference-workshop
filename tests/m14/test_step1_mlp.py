import jax
import jax.numpy as jnp
import numpy as np

from workshop.m14_vae import init_mlp, mlp
from tests._util import key


def test_step1_shapes_and_init_scale():
    params = init_mlp(key(80), [64, 32, 8])
    assert [p["w"].shape for p in params] == [(64, 32), (32, 8)]
    assert all(bool(jnp.all(p["b"] == 0)) for p in params)
    lim = np.sqrt(6.0 / (64 + 32))
    assert float(jnp.abs(params[0]["w"]).max()) <= lim
    assert float(jnp.abs(params[0]["w"]).max()) > 0.8 * lim


def test_step1_forward_matches_hand_computation_and_is_linear_on_output():
    params = init_mlp(key(81), [3, 5, 2])
    x = jax.random.normal(key(82), (7, 3))
    h = jnp.tanh(x @ params[0]["w"] + params[0]["b"])
    ref = h @ params[1]["w"] + params[1]["b"]
    np.testing.assert_allclose(mlp(params, x), ref, rtol=1e-12)
    assert mlp(params, x).shape == (7, 2)
    # Modules 15a and 15b reuse this network; 15b passes a different activation.
    h = jax.nn.silu(x @ params[0]["w"] + params[0]["b"])
    np.testing.assert_allclose(mlp(params, x, activation=jax.nn.silu), h @ params[1]["w"] + params[1]["b"], rtol=1e-12)
