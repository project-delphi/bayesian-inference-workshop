import jax
import jax.numpy as jnp
import numpy as np

from workshop.m14_vae import elbo, elbo_analytic_kl, init_params
from tests.m14._common import L, X
from tests._util import key, mc_close


def test_step3_two_elbos_agree_in_expectation():
    params = init_params(key(100), 64, L)
    xb = X[:200]
    keys = jax.random.split(key(101), 400)
    mc = jax.vmap(lambda k: elbo(k, params, xb, L))(keys)
    an = jax.vmap(lambda k: elbo_analytic_kl(k, params, xb, L))(keys)
    mc_close(mc.mean(), an.mean(), jnp.sqrt(mc.var() / 400 + an.var() / 400), msg="ELBO estimators")
    # the analytic-KL estimator has strictly lower variance
    assert float(an.var()) < float(mc.var())


def test_step3_elbo_is_per_datum_and_differentiable():
    params = init_params(key(102), 64, L)
    e_small = elbo(key(103), params, X[:50], L)
    assert e_small.shape == ()
    assert -200 < float(e_small) < 0
    g = jax.grad(lambda p: elbo_analytic_kl(key(104), p, X[:50], L))(params)
    assert jax.tree_util.tree_structure(g) == jax.tree_util.tree_structure(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(g))
    assert float(jnp.abs(g["enc"][0]["w"]).max()) > 0
