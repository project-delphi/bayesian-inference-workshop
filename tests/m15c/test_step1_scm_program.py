import jax
import jax.numpy as jnp
import numpy as np

from workshop.m12_handlers import log_density, seed, trace
from workshop.m15c_scm import ENDOGENOUS, NOISE_SITES, ad_spend_scm, deterministic, run_scm, structural_matrix
from tests.m15c._common import SCM, structural_matrix_reference
from tests._util import key, mc_close


def test_step1_deterministic_is_recorded_and_ignored_by_log_density():
    def model():
        x = deterministic("x", jnp.asarray(3.0))
        return x

    assert float(deterministic("free", jnp.asarray(1.5))) == 1.5  # no handlers: pass-through
    tr = trace(model).get_trace()
    assert tr["x"]["type"] == "deterministic" and tr["x"]["is_observed"] and tr["x"]["fn"] is None
    lp, tr2 = log_density(ad_spend_scm, {"S": 0.1, "U_A": 0.2, "U_T": -0.3, "U_C": 0.4}, SCM)
    assert {n for n, s in tr2.items() if s["type"] == "deterministic"} == {"A", "T", "C"}
    # only the four noise densities contribute
    ref = sum(float(tr2[n]["fn"].log_prob(tr2[n]["value"])) for n in NOISE_SITES)
    np.testing.assert_allclose(lp, ref, rtol=1e-12)


def test_step1_trace_satisfies_structural_equations():
    tr = trace(seed(ad_spend_scm, key(300))).get_trace(SCM)
    assert set(tr) == set(NOISE_SITES) | {"A", "T", "C"}
    v = {n: float(tr[n]["value"]) for n in tr}
    np.testing.assert_allclose(v["A"], SCM.a_s * v["S"] + v["U_A"], rtol=1e-12)
    np.testing.assert_allclose(v["T"], SCM.b_s * v["S"] + SCM.b_a * v["A"] + v["U_T"], rtol=1e-12)
    np.testing.assert_allclose(v["C"], SCM.c_t * v["T"] + SCM.c_s * v["S"] + v["U_C"], rtol=1e-12)


def test_step1_structural_matrix_and_sample_covariance():
    M, sd = structural_matrix(SCM)
    M_ref, sd_ref = structural_matrix_reference(SCM)
    np.testing.assert_allclose(M, M_ref, rtol=1e-12)
    np.testing.assert_allclose(sd, sd_ref, rtol=1e-12)
    n = 20_000
    draws = run_scm(ad_spend_scm, key(301), n, SCM)
    X = jnp.stack([draws[k] for k in ENDOGENOUS], axis=1)
    assert X.shape == (n, 4)
    cov_ref = M_ref @ np.diag(sd_ref**2) @ M_ref.T
    mc_close(X.mean(0), np.zeros(4), np.sqrt(np.diag(cov_ref) / n), msg="SCM means")
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / (n - 1)
    se = np.sqrt(((Xc[:, :, None] * Xc[:, None, :]) ** 2).mean(0) / n)
    mc_close(cov, cov_ref, se, msg="SCM covariance")
