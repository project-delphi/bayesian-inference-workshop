import jax
import jax.numpy as jnp
import numpy as np

from workshop.m08_bbvi import MeanFieldGaussian, elbo_estimate
from workshop.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob
from workshop.m15a_flows import flow_elbo, flow_sample, tail_mass
from tests._util import key
from tests.m15a._common import flow_fit, mean_field_fit, reference

N = 20_000


def test_step4_reference_has_funnel_tail():
    ref = reference()
    assert ref.shape == (4000, UPLIFT_DIM)
    assert bool(jnp.all(jnp.isfinite(ref)))
    tail = float(tail_mass(ref))
    assert 0.15 < tail < 0.45, tail
    assert float(ref[:, 1].std()) > 0.8


def test_step4_flow_elbo_beats_mean_field():
    fp, ftr = flow_fit()
    mp, _ = mean_field_fit()
    assert bool(jnp.all(jnp.isfinite(ftr)))
    e_flow = float(flow_elbo(key(1510), fp, uplift_centered_log_prob, N))
    e_mf = float(elbo_estimate(key(1511), mp, MeanFieldGaussian(UPLIFT_DIM), uplift_centered_log_prob, N))
    assert e_flow > e_mf + 0.5, (e_flow, e_mf)


def test_step4_flow_captures_log_tau_dispersion_and_tail():
    fp, _ = flow_fit()
    mp, _ = mean_field_fit()
    ref = reference()
    xf, _ = flow_sample(key(1512), fp, N)
    xm = MeanFieldGaussian(UPLIFT_DIM).sample(key(1513), mp, N)
    sd_ref, sd_f, sd_m = (float(a[:, 1].std()) for a in (ref, xf, xm))
    assert abs(sd_f - sd_ref) < abs(sd_m - sd_ref) - 0.2, (sd_ref, sd_f, sd_m)
    t_ref, t_f, t_m = (float(tail_mass(a)) for a in (ref, xf, xm))
    assert abs(t_f - t_ref) < abs(t_m - t_ref) - 0.1, (t_ref, t_f, t_m)
