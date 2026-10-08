import jax.numpy as jnp
import numpy as np
import scipy.integrate
import scipy.stats

from workshop.data import ab_test_conversions
from workshop.m05_conjugacy import BetaBernoulli, posterior_update, prob_b_beats_a, sequential_update
from tests._util import key


def test_step5_sequential_equals_batch():
    ab = ab_test_conversions()
    xs = jnp.asarray(ab.a, dtype=float)
    prior = BetaBernoulli.from_beta(1.0, 1.0)
    final, chis, nus = sequential_update(prior, xs)
    batch = posterior_update(prior, xs)
    np.testing.assert_allclose(final.chi, batch.chi)
    np.testing.assert_allclose(final.nu, batch.nu)
    assert chis.shape == (xs.shape[0], 1) and nus.shape == (xs.shape[0],)
    np.testing.assert_allclose(chis[-1], batch.chi)
    assert bool(jnp.all(jnp.diff(nus) == 1.0))


def test_step5_prob_b_beats_a_matches_quadrature():
    ab = ab_test_conversions()
    prior = BetaBernoulli.from_beta(1.0, 1.0)
    pa = posterior_update(prior, jnp.asarray(ab.a, dtype=float))
    pb = posterior_update(prior, jnp.asarray(ab.b, dtype=float))
    est = prob_b_beats_a(pa, pb, key(120), 400_000)
    aa, ba = (float(v) for v in pa.beta_params())
    ab_, bb = (float(v) for v in pb.beta_params())
    ref, _ = scipy.integrate.quad(lambda p: scipy.stats.beta(ab_, bb).pdf(p) * scipy.stats.beta(aa, ba).cdf(p), 0, 1)
    assert abs(float(est) - ref) < 4 * np.sqrt(ref * (1 - ref) / 400_000) + 1e-3
