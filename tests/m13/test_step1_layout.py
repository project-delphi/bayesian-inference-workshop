import jax
import jax.numpy as jnp
import numpy as np

from workshop.m08_bbvi import Exp
from workshop.m12_handlers import uplift_model
from workshop.m13_ppl_inference import enzyme_model, flatten, unconstrain, unflatten
from tests.m13._common import S, SIGMA, V, Y
from tests._util import key


def test_step1_spec_layout_for_uplift_model():
    spec = unconstrain(uplift_model, Y, SIGMA)
    assert spec.names == ("mu", "tau", "eta")
    assert spec.shapes == ((), (), (8,))
    assert spec.dim == 10
    assert [(s.start, s.stop) for s in spec.slices] == [(0, 1), (1, 2), (2, 10)]
    assert isinstance(spec.bijectors[1], Exp)


def test_step1_flatten_unflatten_roundtrip_and_positivity():
    spec = unconstrain(enzyme_model, S, V)
    z = jax.random.normal(key(40), (spec.dim,))
    vals = unflatten(z, spec)
    assert set(vals) == {"Vmax", "Km", "sigma"}
    assert all(float(v) > 0 for v in vals.values())
    np.testing.assert_allclose(flatten(vals, spec), z, rtol=1e-12)
    np.testing.assert_allclose(vals["Vmax"], jnp.exp(z[0]))
