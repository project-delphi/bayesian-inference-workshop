import jax.numpy as jnp
import numpy as np

from workshop.m15c_scm import abduct_exact, counterfactual, counterfactual_closed_form, run_with, ad_spend_scm
from tests.m15c._common import DATA, SCM, row


def test_step3_abduction_recovers_true_noise_for_every_row():
    obs = {k: DATA[k] for k in ("S", "A", "T", "C")}
    u = abduct_exact(SCM, obs)
    for k in ("U_A", "U_T", "U_C"):
        np.testing.assert_allclose(u[k], DATA[k], atol=1e-10)
    np.testing.assert_allclose(u["S"], DATA["S"])


def test_step3_run_with_reproduces_the_observed_week():
    r = row(7)
    world = run_with(ad_spend_scm, abduct_exact(SCM, r), SCM)
    for k in ("S", "A", "T", "C"):
        np.testing.assert_allclose(world[k], r[k], atol=1e-10)


def test_step3_counterfactual_matches_closed_form_for_halved_spend():
    r = row(11)
    a_new = r["A"] / 2
    cf = counterfactual(SCM, r, {"A": a_new})
    np.testing.assert_allclose(cf["A"], a_new)
    np.testing.assert_allclose(cf["C"], counterfactual_closed_form(SCM, r, a_new), rtol=1e-12)
    # the factual noise is kept: S and U_C of the counterfactual world equal the abducted ones
    np.testing.assert_allclose(cf["S"], r["S"])
    np.testing.assert_allclose(cf["U_C"], r["U_C"], atol=1e-10)
    # effect on C is c_t b_a (a_new - A), independent of the noise
    np.testing.assert_allclose(cf["C"] - r["C"], SCM.c_t * SCM.b_a * (a_new - r["A"]), rtol=1e-10)
