import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m12_handlers import seed, trace
from workshop.m15c_scm import abduct_exact, do, exact_counterfactual_C, probability_below, run_with, twin_network, twin_posterior
from tests.m15c._common import SCM, row
from tests._util import key

OBS = {"A": row(3)["A"], "T": row(3)["T"]}
A_NEW = OBS["A"] / 2


def test_step5_twin_network_shares_noise_and_reproduces_factual_world():
    r = row(5)
    world = run_with(do(twin_network, {"A_cf": 0.0}), abduct_exact(SCM, r), SCM)
    for k in ("A", "T", "C"):
        np.testing.assert_allclose(world[k], r[k], atol=1e-10)
    assert float(world["A_cf"]) == 0.0
    np.testing.assert_allclose(world["T_cf"], SCM.b_s * r["S"] + r["U_T"], atol=1e-10)
    # without intervention the two worlds coincide
    tr = trace(seed(twin_network, key(330))).get_trace(SCM)
    for k in ("A", "T", "C"):
        np.testing.assert_allclose(tr[k]["value"], tr[k + "_cf"]["value"])


def test_step5_twin_posterior_factual_matches_observations():
    post = twin_posterior(SCM, OBS, A_NEW, key(331))
    np.testing.assert_allclose(post["A"], OBS["A"], atol=1e-8)
    np.testing.assert_allclose(post["T"], OBS["T"], atol=1e-8)
    assert bool(jnp.all(post["A_cf"] == A_NEW))
    mean, sd = exact_counterfactual_C(SCM, OBS, A_NEW)
    assert abs(float(post["C_cf"].mean()) - float(mean)) < 3 * float(sd) / np.sqrt(200)


def test_step5_probability_below_threshold_matches_normal_cdf():
    mean, sd = exact_counterfactual_C(SCM, OBS, A_NEW)
    threshold = float(mean) + 0.5 * float(sd)
    p = probability_below(SCM, OBS, A_NEW, threshold, key(332))
    ref = scipy.stats.norm(float(mean), float(sd)).cdf(threshold)
    assert abs(float(p) - ref) < 0.06
