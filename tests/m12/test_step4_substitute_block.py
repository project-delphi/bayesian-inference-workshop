import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m12_handlers import Normal, block, log_density, param, sample, seed, substitute, trace, two_site_model, uplift_model
from workshop.data import regional_uplift
from tests._util import key

Y, SIGMA = (jnp.asarray(a) for a in regional_uplift())


def test_step4_substitute_overrides_sample_and_param_sites():
    def m():
        w = param("w", jnp.zeros(2))
        z = sample("z", Normal(0.0, 1.0))
        return w, z

    w, z = substitute(m, {"w": jnp.array([1.0, 2.0]), "z": jnp.asarray(0.5)})()
    np.testing.assert_array_equal(w, [1.0, 2.0])
    assert z == 0.5
    tr = trace(substitute(m, {"w": jnp.ones(2), "z": jnp.asarray(0.5)})).get_trace()
    assert not tr["z"]["is_observed"]  # substituted latents stay latent
    assert tr["w"]["type"] == "param"


def test_step4_block_hides_sites_from_outer_trace():
    tr = trace(block(seed(uplift_model, key(20)), hide=["eta", "y"])).get_trace(Y, SIGMA)
    assert list(tr) == ["mu", "tau"]
    tr2 = trace(block(seed(uplift_model, key(20)), expose=["eta"])).get_trace(Y, SIGMA)
    assert list(tr2) == ["eta"]
    tr3 = trace(block(seed(uplift_model, key(20)))).get_trace(Y, SIGMA)
    assert list(tr3) == []
    # blocked sites are still executed: inner trace sees them
    inner = trace(seed(uplift_model, key(20)))
    trace(block(inner, hide=["eta"])).get_trace(Y, SIGMA)
    assert "eta" in inner.trace


def test_step4_log_density_two_site_model_matches_hand_sum():
    x = jnp.array([0.3, -0.2, 0.9])
    mu = jnp.asarray(0.4)
    lp, tr = log_density(two_site_model, {"mu": mu}, x)
    ref = scipy.stats.norm(0, 1).logpdf(0.4) + scipy.stats.norm(0.4, 0.5).logpdf(np.asarray(x)).sum()
    np.testing.assert_allclose(lp, ref, rtol=1e-12)
    assert list(tr) == ["mu", "x"]
