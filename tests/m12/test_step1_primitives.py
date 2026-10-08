import jax
import jax.numpy as jnp
import numpy as np
import pytest

from workshop.m12_handlers import Bernoulli, Messenger, Normal, _HANDLER_STACK, apply_stack, param, sample


def test_step1_sample_without_key_raises_naming_site():
    with pytest.raises(RuntimeError, match="needs a PRNG key"):
        sample("alpha", Normal(0.0, 1.0))


def test_step1_observed_sample_and_param_pass_through():
    x = jnp.array([1.0, 2.0])
    assert sample("x", Normal(0.0, 1.0), obs=x) is x
    np.testing.assert_array_equal(param("w", jnp.ones(3)), jnp.ones(3))


def test_step1_apply_stack_order_and_stop():
    log = []

    class Recorder(Messenger):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def process_message(self, msg):
            log.append(("down", self.tag))
            if self.tag == "inner" and msg["name"] == "blocked":
                msg["stop"] = True

        def postprocess_message(self, msg):
            log.append(("up", self.tag))

    with Recorder("outer"), Recorder("inner"):
        msg = {"type": "param", "name": "p", "fn": None, "value": None, "is_observed": False, "key": None, "sample_shape": (), "stop": False, "init_value": jnp.asarray(2.0)}
        apply_stack(msg)
        assert msg["value"] == 2.0
        assert log == [("down", "inner"), ("down", "outer"), ("up", "outer"), ("up", "inner")]
        log.clear()
        msg2 = dict(msg, name="blocked", value=None)
        apply_stack(msg2)
        assert log == [("down", "inner"), ("up", "inner")]
    assert len(_HANDLER_STACK) == 0


def test_step1_stack_is_popped_on_exception():
    class Boom(Messenger):
        pass

    with pytest.raises(ZeroDivisionError):
        with Boom():
            1 / 0
    assert len(_HANDLER_STACK) == 0
    # and the machinery still works afterwards: an observed site under a handler
    x = jnp.array([1.0])
    with Boom():
        assert sample("x", Normal(0.0, 1.0), obs=x) is x
    assert len(_HANDLER_STACK) == 0


def test_step1_default_sample_uses_key_and_shape():
    class GiveKey(Messenger):
        def process_message(self, msg):
            msg["key"] = jax.random.key(3)

    with GiveKey():
        v = sample("z", Normal(jnp.zeros(2), 1.0), sample_shape=(5,))
    assert v.shape == (5, 2)
    with GiveKey():
        b = sample("b", Bernoulli(logits=jnp.zeros(4)))
    assert b.shape == (4,) and set(np.unique(np.asarray(b))) <= {0.0, 1.0}
