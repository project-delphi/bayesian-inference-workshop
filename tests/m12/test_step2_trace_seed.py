import jax
import jax.numpy as jnp
import numpy as np
import pytest

from workshop.data import regional_uplift
from workshop.m12_handlers import Normal, sample, seed, trace, uplift_model
from tests._util import key

Y, SIGMA = (jnp.asarray(a) for a in regional_uplift())


def test_step2_trace_records_sites_in_order_with_metadata():
    tr = trace(seed(uplift_model, key(1))).get_trace(Y, SIGMA)
    assert list(tr) == ["mu", "tau", "eta", "y"]
    assert tr["y"]["is_observed"] and not tr["mu"]["is_observed"]
    assert tr["eta"]["value"].shape == (8,)
    np.testing.assert_array_equal(tr["y"]["value"], Y)
    assert tr["tau"]["value"] > 0
    assert all(s["type"] == "sample" for s in tr.values())


def test_step2_seed_is_deterministic_and_site_specific():
    t1 = trace(seed(uplift_model, key(2))).get_trace(Y, SIGMA)
    t2 = trace(seed(uplift_model, key(2))).get_trace(Y, SIGMA)
    t3 = trace(seed(uplift_model, key(3))).get_trace(Y, SIGMA)
    for name in ("mu", "tau", "eta"):
        np.testing.assert_array_equal(t1[name]["value"], t2[name]["value"])
        assert not np.allclose(t1[name]["value"], t3[name]["value"])
    # different sites get different keys: two N(0,1) sites must not be identical draws
    def two():
        a = sample("a", Normal(0.0, 1.0))
        b = sample("b", Normal(0.0, 1.0))
        return a, b

    a, b = seed(two, key(4))()
    assert not np.isclose(a, b)


def test_step2_duplicate_site_name_is_an_error():
    def bad():
        sample("a", Normal(0.0, 1.0))
        sample("a", Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="duplicate"):
        trace(seed(bad, key(5))).get_trace()
