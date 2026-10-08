"""Module 12 — Execution traces and effect handlers.

A probabilistic program is an ordinary Python function that calls `sample(name, dist)`
and `param(name, init)`. Each call builds a *message* and passes it down a stack of
*handlers* (Messengers). A handler may fill in the value (replay, condition,
substitute), supply a PRNG key (seed), record the message (trace) or hide it from outer
handlers (block). The default behaviour, used when no handler supplies a value, is to
draw from the distribution with the key provided by `seed`.

This is the mechanism behind Pyro and NumPyro, reduced to about 200 lines. Modules 13
and 14 run the Day 2 VI engine and the Day 3 HMC sampler through it.

Message dict keys:
    type         "sample" or "param"
    name         site name (unique within one execution)
    fn           the Distribution (sample sites) or None (param sites)
    value        the value at the site, or None until resolved
    is_observed  True when the value came from `obs=` or from `condition`
    key          PRNG key for this site (set by `seed`), or None
    sample_shape extra leading sample dimensions requested at the site
    stop         True stops propagation to handlers further out (set by `block`)
    init_value   param sites only: the initial value
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .m08_bbvi import Bijector, Chain, Exp, Identity, Sigmoid

Array = jax.Array
Message = dict[str, Any]
Trace = "OrderedDict[str, Message]"

_LOG_2PI = jnp.log(2 * jnp.pi)


# ======================================================================================
# Distributions (provided, not stubbed). log_prob is elementwise; the caller sums.
# `support` is the bijector from R to the support, used by Module 13 to unconstrain.
# ======================================================================================
class Affine(Bijector):
    """y = loc + scale * x (elementwise), scale > 0."""

    def __init__(self, loc, scale):
        self.loc, self.scale = jnp.asarray(loc), jnp.asarray(scale)

    def forward(self, x):
        return self.loc + self.scale * x

    def inverse(self, y):
        return (y - self.loc) / self.scale

    def log_det_jacobian(self, x):
        return jnp.sum(jnp.broadcast_to(jnp.log(self.scale), jnp.shape(x)))


class Distribution:
    support: Bijector | None = Identity()
    _params: tuple = ()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(*[jnp.shape(p) for p in self._params]) if self._params else ()

    def log_prob(self, x: Array) -> Array:
        raise NotImplementedError

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc, self.scale = jnp.asarray(loc), jnp.asarray(scale)
        self._params = (self.loc, self.scale)

    def log_prob(self, x):
        return -0.5 * ((x - self.loc) / self.scale) ** 2 - jnp.log(self.scale) - 0.5 * _LOG_2PI

    def sample(self, key, sample_shape=()):
        return self.loc + self.scale * jax.random.normal(key, tuple(sample_shape) + self.batch_shape)


class HalfNormal(Distribution):
    support = Exp()

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)
        self._params = (self.scale,)

    def log_prob(self, x):
        return jnp.log(2.0) - 0.5 * _LOG_2PI - jnp.log(self.scale) - 0.5 * (x / self.scale) ** 2

    def sample(self, key, sample_shape=()):
        return self.scale * jnp.abs(jax.random.normal(key, tuple(sample_shape) + self.batch_shape))


class LogNormal(Distribution):
    support = Exp()

    def __init__(self, loc, scale):
        self.loc, self.scale = jnp.asarray(loc), jnp.asarray(scale)
        self._params = (self.loc, self.scale)

    def log_prob(self, x):
        lx = jnp.log(x)
        return -0.5 * ((lx - self.loc) / self.scale) ** 2 - jnp.log(self.scale) - 0.5 * _LOG_2PI - lx

    def sample(self, key, sample_shape=()):
        return jnp.exp(self.loc + self.scale * jax.random.normal(key, tuple(sample_shape) + self.batch_shape))


class Gamma(Distribution):
    """Shape `concentration`, rate `rate`."""

    support = Exp()

    def __init__(self, concentration, rate):
        self.concentration, self.rate = jnp.asarray(concentration), jnp.asarray(rate)
        self._params = (self.concentration, self.rate)

    def log_prob(self, x):
        a, b = self.concentration, self.rate
        return a * jnp.log(b) - gammaln(a) + (a - 1) * jnp.log(x) - b * x

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape
        return jax.random.gamma(key, jnp.broadcast_to(self.concentration, shape)) / self.rate


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low, self.high = jnp.asarray(low), jnp.asarray(high)
        self._params = (self.low, self.high)
        self.support = Chain(Sigmoid(), Affine(self.low, self.high - self.low))

    def log_prob(self, x):
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, -jnp.log(self.high - self.low), -jnp.inf)

    def sample(self, key, sample_shape=()):
        return jax.random.uniform(key, tuple(sample_shape) + self.batch_shape, minval=self.low, maxval=self.high)


class Bernoulli(Distribution):
    """Parameterised by logits. Discrete: no unconstraining bijector (support = None)."""

    support = None

    def __init__(self, logits):
        self.logits = jnp.asarray(logits)
        self._params = (self.logits,)

    def log_prob(self, x):
        # x log s(l) + (1 - x) log s(-l), computed stably
        return x * jax.nn.log_sigmoid(self.logits) + (1 - x) * jax.nn.log_sigmoid(-self.logits)

    def sample(self, key, sample_shape=()):
        return jax.random.bernoulli(key, jax.nn.sigmoid(self.logits), tuple(sample_shape) + self.batch_shape).astype(jnp.float64)


# ======================================================================================
# Step 1: messages, the handler stack, and the two primitives
# ======================================================================================
_HANDLER_STACK: list["Messenger"] = []


class Messenger:
    """Base handler. A context manager that sits on the global stack while active, and a
    callable that wraps `fn` so `handler(fn)(*args)` runs `fn` inside the handler."""

    def __init__(self, fn: Callable | None = None):
        self.fn = fn

    def __enter__(self):
        _HANDLER_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Pop unconditionally, including when the body raised; otherwise one failed
        # execution leaves a stale handler on the stack for the rest of the process.
        popped = _HANDLER_STACK.pop()
        assert popped is self, "handler stack corrupted"

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)

    def process_message(self, msg: Message) -> None:
        """Called on the way down the stack (innermost handler first)."""

    def postprocess_message(self, msg: Message) -> None:
        """Called on the way back up (innermost handler last)."""


def _new_message(type_: str, name: str, **fields) -> Message:
    msg: Message = {
        "type": type_,
        "name": name,
        "fn": None,
        "value": None,
        "is_observed": False,
        "key": None,
        "sample_shape": (),
        "stop": False,
        "init_value": None,
    }
    msg.update(fields)
    return msg


def apply_stack(msg: Message) -> Message:
    """Send `msg` down the handler stack, apply the default behaviour if no handler set
    a value, then send it back up.

    Down: iterate handlers from the top of the stack (innermost) outward, calling
    `process_message`; stop early if `msg["stop"]` becomes True. Default: a sample site
    with no value draws `fn.sample(key, sample_shape)` and raises a RuntimeError naming
    the site if `key` is None; a param site with no value takes `init_value`.
    Up: call `postprocess_message` on the handlers that saw the message, in reverse.
    """
    # [m12 step 1]
    seen: list[Messenger] = []
    for handler in reversed(_HANDLER_STACK):
        handler.process_message(msg)
        seen.append(handler)
        if msg["stop"]:
            break
    if msg["value"] is None:
        if msg["type"] == "sample":
            if msg["key"] is None:
                raise RuntimeError(
                    f"sample site '{msg['name']}' needs a PRNG key: wrap the model in seed(model, key) "
                    "or supply its value with condition/substitute/replay"
                )
            msg["value"] = msg["fn"].sample(msg["key"], msg["sample_shape"])
        else:
            msg["value"] = msg["init_value"]
    for handler in reversed(seen):
        handler.postprocess_message(msg)
    return msg


def sample(name: str, fn: Distribution, obs: Array | None = None, sample_shape: tuple[int, ...] = ()) -> Array:
    """Primitive: a random variable named `name` with distribution `fn`. With `obs`, the
    site is observed and its value is `obs`. With no handlers on the stack and no obs,
    draw from fn requires a key, so it raises; use seed(...)."""
    # [m12 step 1]
    msg = _new_message("sample", name, fn=fn, value=obs, is_observed=obs is not None, sample_shape=tuple(sample_shape))
    if not _HANDLER_STACK and obs is not None:
        return obs
    return apply_stack(msg)["value"]


def param(name: str, init_value) -> Any:
    """Primitive: a learnable parameter site. `init_value` may be an array or a pytree of
    arrays (a list of layers, say). Returns `init_value` unless a handler (substitute)
    overrides it."""
    # [m12 step 1]
    msg = _new_message("param", name, init_value=init_value)
    if not _HANDLER_STACK:
        return msg["init_value"]
    return apply_stack(msg)["value"]


# ======================================================================================
# Step 2: trace and seed
# ======================================================================================
class trace(Messenger):
    """Record every message that reaches this handler, keyed by site name.

    `get_trace(*args, **kwargs)` runs fn and returns the OrderedDict of messages. A
    repeated site name is an error."""

    def __enter__(self):
        self.trace: "OrderedDict[str, Message]" = OrderedDict()
        return super().__enter__()

    def postprocess_message(self, msg):
        # [m12 step 2]
        if msg["name"] in self.trace:
            raise ValueError(f"duplicate site name '{msg['name']}' in one execution")
        self.trace[msg["name"]] = dict(msg)

    def get_trace(self, *args, **kwargs):
        # [m12 step 2]
        self(*args, **kwargs)
        return self.trace


class seed(Messenger):
    """Give each unresolved sample site a fresh key: the i-th such site in execution
    order gets jax.random.fold_in(key, i). Deterministic for a fixed key and program."""

    def __init__(self, fn: Callable | None = None, key: Array | None = None):
        super().__init__(fn)
        self.key = key

    def __enter__(self):
        self.counter = 0
        return super().__enter__()

    def process_message(self, msg):
        # [m12 step 2]
        if msg["type"] == "sample" and msg["value"] is None and msg["key"] is None:
            msg["key"] = jax.random.fold_in(self.key, self.counter)
            self.counter += 1


# ======================================================================================
# Step 3: condition and replay
# ======================================================================================
class condition(Messenger):
    """Fix sample sites named in `data` to the given values and mark them observed."""

    def __init__(self, fn: Callable | None = None, data: dict[str, Array] | None = None):
        super().__init__(fn)
        self.data = data or {}

    def process_message(self, msg):
        # [m12 step 3]
        if msg["type"] == "sample" and msg["name"] in self.data:
            msg["value"] = self.data[msg["name"]]
            msg["is_observed"] = True


class replay(Messenger):
    """Reuse the values of a previous trace at sample sites with the same name. Observed
    sites in the current execution keep their own values."""

    def __init__(self, fn: Callable | None = None, guide_trace: "OrderedDict[str, Message]" | None = None):
        super().__init__(fn)
        self.guide_trace = guide_trace or OrderedDict()

    def process_message(self, msg):
        # [m12 step 3]
        if msg["type"] == "sample" and not msg["is_observed"] and msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


# ======================================================================================
# Step 4: substitute and block
# ======================================================================================
class substitute(Messenger):
    """Set the value of sample *and* param sites from `values` (latent sites stay
    unobserved, which is what log_density needs)."""

    def __init__(self, fn: Callable | None = None, values: dict[str, Array] | None = None):
        super().__init__(fn)
        self.values = values or {}

    def process_message(self, msg):
        # [m12 step 4]
        if msg["name"] in self.values:
            msg["value"] = self.values[msg["name"]]


class block(Messenger):
    """Hide sites from handlers further out. With `hide`, those names are hidden; with
    `expose`, everything except those names is hidden; with neither, hide everything.
    Hiding is done by setting msg["stop"] = True, so outer handlers never see it."""

    def __init__(self, fn: Callable | None = None, hide: list[str] | None = None, expose: list[str] | None = None):
        super().__init__(fn)
        self.hide, self.expose = hide, expose

    def _hidden(self, name: str) -> bool:
        if self.expose is not None:
            return name not in self.expose
        if self.hide is not None:
            return name in self.hide
        return True

    def process_message(self, msg):
        # [m12 step 4]
        if self._hidden(msg["name"]):
            msg["stop"] = True


# ======================================================================================
# Step 5: the log joint density
# ======================================================================================
def log_density(model: Callable, values: dict[str, Array], *args, **kwargs) -> tuple[Array, "OrderedDict[str, Message]"]:
    """Run `model` with latent sites fixed to `values` (and params overridden where
    given), record the trace, and return (sum over sample sites of sum(log_prob(value)),
    trace). Every latent site must be in `values`; no key is needed. jit-able and
    differentiable with respect to `values`."""
    # [m12 step 5]
    tr = trace(substitute(model, values)).get_trace(*args, **kwargs)
    lp = jnp.zeros(())
    for site in tr.values():
        if site["type"] == "sample":
            lp = lp + jnp.sum(site["fn"].log_prob(site["value"]))
    return lp, tr


# ======================================================================================
# Example models used in tests and later modules
# ======================================================================================
def uplift_model(y: Array, sigma: Array) -> None:
    """Hierarchical regional uplift, non-centred (Module 9's target, written as a program)."""
    mu = sample("mu", Normal(0.0, 5.0))
    tau = sample("tau", HalfNormal(5.0))
    eta = sample("eta", Normal(jnp.zeros(8), 1.0))
    theta = mu + tau * eta
    sample("y", Normal(theta, sigma), obs=y)


def two_site_model(x: Array) -> None:
    """mu ~ N(0, 1); x_i ~ N(mu, 0.5^2)."""
    mu = sample("mu", Normal(0.0, 1.0))
    sample("x", Normal(mu, 0.5), obs=x)
