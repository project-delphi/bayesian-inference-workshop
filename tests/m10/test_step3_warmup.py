import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import hmc
from workshop.m10_adaptation import warmup, warmup_schedule
from tests._util import key


def test_step3_schedule_layout():
    is_slow, end = warmup_schedule(1000)
    assert is_slow.shape == (1000,) and end.shape == (1000,)
    assert not is_slow[:150].any() and not is_slow[900:].any()
    assert is_slow[150:900].all()
    ends = np.flatnonzero(end)
    # windows 25, 50, 100, then the fourth extended to the end of the slow phase (Stan's rule)
    assert ends.tolist() == [174, 224, 324, 899]
    s, e = warmup_schedule(10)
    assert not s.any() and not e.any()


def test_step3_adapts_mass_and_step_on_badly_scaled_gaussian():
    var = jnp.array([1.0, 100.0])
    lp = lambda z: -0.5 * jnp.sum(z**2 / var)
    adapted, infos = jax.jit(lambda k: warmup(lp, k, jnp.zeros(2), 1000))(key(250))
    ratio = adapted.inv_mass / var
    assert bool(jnp.all((ratio > 1 / 1.5) & (ratio < 1.5))), ratio
    samples, info = hmc(lp, key(251), adapted.q, 2000, float(adapted.step_size), 8, adapted.inv_mass)
    acc = float(info.accept_prob.mean())
    assert 0.6 < acc < 0.98, acc
    np.testing.assert_allclose(samples.var(0), var, rtol=0.25)
