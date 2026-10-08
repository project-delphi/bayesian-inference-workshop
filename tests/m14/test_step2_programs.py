import jax
import jax.numpy as jnp
import numpy as np

from workshop.m12_handlers import seed, trace
from workshop.m14_vae import init_params, vae_guide, vae_model
from tests.m14._common import L, X
from tests._util import key


def test_step2_model_trace_has_local_latents_and_observed_x():
    params = init_params(key(90), 64, L)
    xb = X[:16]
    tr = trace(seed(vae_model, key(91))).get_trace(params, xb, L)
    assert list(tr) == ["dec", "z", "x"]
    assert tr["dec"]["type"] == "param"
    assert tr["z"]["value"].shape == (16, L) and not tr["z"]["is_observed"]
    assert tr["x"]["is_observed"]
    np.testing.assert_array_equal(tr["x"]["value"], xb)
    assert tr["x"]["fn"].logits.shape == (16, 64)


def test_step2_guide_is_amortised_and_positive_scale():
    params = init_params(key(92), 64, L)
    xb = X[:16]
    tr = trace(seed(vae_guide, key(93))).get_trace(params, xb, L)
    assert list(tr) == ["enc", "z"]
    q = tr["z"]["fn"]
    assert q.loc.shape == (16, L) and q.scale.shape == (16, L)
    assert bool(jnp.all(q.scale > 0))
    # amortisation: identical inputs give identical variational parameters
    tr2 = trace(seed(vae_guide, key(94))).get_trace(params, jnp.stack([xb[0], xb[0]]), L)
    np.testing.assert_allclose(tr2["z"]["fn"].loc[0], tr2["z"]["fn"].loc[1])
