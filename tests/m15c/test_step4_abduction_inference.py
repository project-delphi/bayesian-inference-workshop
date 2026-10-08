import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from workshop.m15c_scm import abduction_model, counterfactual_posterior, exact_counterfactual_C
from workshop.m13_ppl_inference import unconstrain
from tests.m15c._common import SCM, row
from tests._util import key

OBS = {"A": row(3)["A"], "T": row(3)["T"]}
A_NEW = OBS["A"] / 2


def test_step4_abduction_model_has_two_latent_sites():
    spec = unconstrain(abduction_model, SCM, OBS)
    assert set(spec.names) == {"S", "U_C"} and spec.dim == 2


def test_step4_exact_conditional_matches_regression_on_simulation():
    # Independent check: in a linear-Gaussian system E[C_cf | A, T] is the least-squares
    # regression of C_cf on (1, A, T), and Var[C_cf | A, T] is its residual variance.
    rng = np.random.default_rng(0)
    n = 400_000
    s = rng.normal(size=n)
    u_a = SCM.sd_a * rng.normal(size=n)
    u_t = SCM.sd_t * rng.normal(size=n)
    u_c = SCM.sd_c * rng.normal(size=n)
    a = SCM.a_s * s + u_a
    t = SCM.b_s * s + SCM.b_a * a + u_t
    a_new = 0.5 * a
    c_cf = SCM.c_t * (SCM.b_s * s + SCM.b_a * a_new + u_t) + SCM.c_s * s + u_c
    X = np.column_stack([np.ones(n), a, t])
    beta, *_ = np.linalg.lstsq(X, c_cf, rcond=None)
    resid_sd = np.std(c_cf - X @ beta)
    mean, sd = exact_counterfactual_C(SCM, OBS, A_NEW)
    pred = beta @ np.array([1.0, float(OBS["A"]), float(OBS["T"])])
    np.testing.assert_allclose(mean, pred, atol=0.01)
    np.testing.assert_allclose(sd, resid_sd, rtol=0.01)


def test_step4_hmc_counterfactual_posterior_matches_exact_gaussian():
    post = counterfactual_posterior(SCM, OBS, {"A": A_NEW}, key(320), method="hmc")
    assert post["C"].shape == (1200,)
    assert bool(jnp.all(post["A"] == A_NEW))
    np.testing.assert_allclose(post["T"], OBS["T"] + SCM.b_a * (A_NEW - OBS["A"]), atol=1e-8)
    mean, sd = exact_counterfactual_C(SCM, OBS, A_NEW)
    assert abs(float(post["C"].mean()) - float(mean)) < 3 * float(sd) / np.sqrt(200)
    assert abs(float(post["C"].std()) / float(sd) - 1.0) < 0.3


def test_step4_svi_counterfactual_posterior_matches_exact_gaussian():
    post = counterfactual_posterior(SCM, OBS, {"A": A_NEW}, key(321), method="svi")
    mean, sd = exact_counterfactual_C(SCM, OBS, A_NEW)
    assert abs(float(post["C"].mean()) - float(mean)) < 0.15 * float(sd)
    assert abs(float(post["C"].std()) / float(sd) - 1.0) < 0.3
