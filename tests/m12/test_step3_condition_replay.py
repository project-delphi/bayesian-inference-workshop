import jax
import jax.numpy as jnp
import numpy as np

from workshop.data import regional_uplift
from workshop.m12_handlers import Normal, condition, replay, sample, seed, trace, uplift_model
from tests._util import key

Y, SIGMA = (jnp.asarray(a) for a in regional_uplift())


def prior_model(sigma):
    mu = sample("mu", Normal(0.0, 5.0))
    tau = sample("tau", Normal(1.0, 0.1))
    sample("y", Normal(mu, sigma))


def test_step3_condition_marks_observed_and_sets_value():
    tr = trace(seed(condition(prior_model, {"y": Y}), key(10))).get_trace(SIGMA)
    assert tr["y"]["is_observed"]
    np.testing.assert_array_equal(tr["y"]["value"], Y)
    assert not tr["mu"]["is_observed"]
    # a different key changes mu but not the conditioned y
    tr2 = trace(seed(condition(prior_model, {"y": Y}), key(11))).get_trace(SIGMA)
    assert not np.isclose(tr["mu"]["value"], tr2["mu"]["value"])
    np.testing.assert_array_equal(tr2["y"]["value"], Y)


def test_step3_replay_reproduces_latents_but_not_observed():
    guide_tr = trace(seed(uplift_model, key(12))).get_trace(Y, SIGMA)
    tr = trace(seed(replay(uplift_model, guide_tr), key(13))).get_trace(Y, SIGMA)
    for name in ("mu", "tau", "eta"):
        np.testing.assert_array_equal(tr[name]["value"], guide_tr[name]["value"])
    np.testing.assert_array_equal(tr["y"]["value"], Y)
    assert tr["y"]["is_observed"]


def test_step3_replay_only_touches_matching_names():
    def guide():
        sample("mu", Normal(3.0, 0.01))

    gtr = trace(seed(guide, key(14))).get_trace()
    tr = trace(seed(replay(uplift_model, gtr), key(15))).get_trace(Y, SIGMA)
    np.testing.assert_array_equal(tr["mu"]["value"], gtr["mu"]["value"])
    assert "tau" in tr and tr["tau"]["value"] > 0
