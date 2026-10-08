import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m08_bbvi import MeanFieldGaussian
from tests._util import key, mc_close


def test_step3_log_prob_and_entropy_match_scipy():
    fam = MeanFieldGaussian(3)
    params = {"loc": jnp.array([0.5, -1.0, 2.0]), "log_scale": jnp.array([0.0, -0.5, 0.7])}
    z = jnp.array([0.1, 0.2, 0.3])
    sd = np.exp(np.asarray(params["log_scale"]))
    ref = scipy.stats.norm(np.asarray(params["loc"]), sd).logpdf(np.asarray(z)).sum()
    np.testing.assert_allclose(fam.log_prob(params, z), ref, rtol=1e-12)
    np.testing.assert_allclose(fam.entropy(params), scipy.stats.norm(0, sd).entropy().sum(), rtol=1e-12)


def test_step3_sample_is_reparameterised_and_has_right_moments():
    fam = MeanFieldGaussian(2)
    params = {"loc": jnp.array([1.0, -1.0]), "log_scale": jnp.array([jnp.log(0.5), jnp.log(2.0)])}
    n = 200_000
    z = fam.sample(key(410), params, n)
    assert z.shape == (n, 2)
    mc_close(z.mean(0), params["loc"], jnp.exp(params["log_scale"]) / np.sqrt(n), msg="mean")
    # gradient of the sample mean w.r.t. loc must be exactly 1 (pathwise)
    g = jax.grad(lambda p: fam.sample(key(410), p, 100).mean())(params)
    np.testing.assert_allclose(g["loc"], 0.5, rtol=1e-12)
