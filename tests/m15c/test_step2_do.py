import jax
import jax.numpy as jnp
import numpy as np

from workshop.m12_handlers import log_density, seed, trace
from workshop.m15c_scm import (
    ad_spend_scm,
    backdoor_slope,
    causal_effect,
    confounding_bias,
    do,
    interventional_samples,
    observational_slope,
)
from tests.m15c._common import DATA, SCM
from tests._util import key, mc_close


def test_step2_do_on_endogenous_node_propagates_downstream():
    tr = trace(seed(do(ad_spend_scm, {"A": 2.0}), key(310))).get_trace(SCM)
    assert tr["A"]["intervened"] and float(tr["A"]["value"]) == 2.0
    v = {n: float(tr[n]["value"]) for n in tr}
    np.testing.assert_allclose(v["T"], SCM.b_s * v["S"] + SCM.b_a * 2.0 + v["U_T"], rtol=1e-12)
    np.testing.assert_allclose(v["C"], SCM.c_t * v["T"] + SCM.c_s * v["S"] + v["U_C"], rtol=1e-12)
    # E[C | do(A = a)] = c_t b_a a since E[S] = 0
    n = 50_000
    d = interventional_samples(SCM, {"A": 2.0}, key(311), n)
    assert bool(jnp.all(d["A"] == 2.0))
    mc_close(d["C"].mean(), SCM.c_t * SCM.b_a * 2.0, d["C"].std() / np.sqrt(n), msg="E[C | do(A)]")


def test_step2_do_on_noise_site_removes_it_from_the_density():
    tr = trace(seed(do(ad_spend_scm, {"U_T": 0.0}), key(312))).get_trace(SCM)
    assert tr["U_T"]["type"] == "deterministic" and tr["U_T"]["fn"] is None
    v = {n: float(tr[n]["value"]) for n in tr}
    np.testing.assert_allclose(v["T"], SCM.b_s * v["S"] + SCM.b_a * v["A"], rtol=1e-12)
    vals = {"S": 0.1, "U_A": 0.2, "U_C": 0.4}
    lp, tr2 = log_density(do(ad_spend_scm, {"U_T": 0.0}), vals, SCM)
    ref = sum(float(tr2[n]["fn"].log_prob(tr2[n]["value"])) for n in ("S", "U_A", "U_C"))
    np.testing.assert_allclose(lp, ref, rtol=1e-12)


def test_step2_observational_slope_is_confounded_and_backdoor_fixes_it():
    a, c, s = DATA["A"], DATA["C"], DATA["S"]
    slope_obs = observational_slope(a, c)
    slope_bd = backdoor_slope(a, c, s)
    effect = causal_effect(SCM)
    bias = confounding_bias(SCM)
    np.testing.assert_allclose(effect, SCM.c_t * SCM.b_a, rtol=1e-12)
    # closed-form slope: effect + bias; sampling error of a slope on 2000 rows is ~0.02
    np.testing.assert_allclose(slope_obs, effect + bias, atol=0.06)
    np.testing.assert_allclose(slope_bd, effect, atol=0.06)
    assert float(bias) > 0.15  # confounding is material here
    assert abs(float(slope_obs) - float(effect)) > 3 * abs(float(slope_bd) - float(effect))
