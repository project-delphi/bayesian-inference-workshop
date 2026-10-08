import jax
import jax.numpy as jnp
import numpy as np

from workshop.m08_bbvi import Chain, Exp, Identity, Sigmoid, Softplus
from tests._util import key

X = jax.random.normal(key(400), (6,)) * 2


def _check(b):
    y = b.forward(X)
    np.testing.assert_allclose(b.inverse(y), X, rtol=1e-9, atol=1e-10)
    J = jax.jacfwd(b.forward)(X)
    np.testing.assert_allclose(b.log_det_jacobian(X), jnp.linalg.slogdet(J)[1], rtol=1e-9, atol=1e-10)


def test_step1_exp_softplus_sigmoid():
    for b in (Exp(), Softplus(), Sigmoid(), Identity()):
        _check(b)


def test_step1_supports():
    assert bool(jnp.all(Exp().forward(X) > 0))
    assert bool(jnp.all(Softplus().forward(X) > 0))
    s = Sigmoid().forward(X)
    assert bool(jnp.all((s > 0) & (s < 1)))


def test_step1_chain():
    b = Chain(Softplus(), Exp())
    _check(b)
    assert b.inverse(b.forward(jnp.array([0.3]))).shape == (1,)


def test_step1_softplus_inverse_is_stable_for_large_values():
    y = jnp.array([1e-8, 1.0, 50.0, 700.0])
    x = Softplus().inverse(y)
    assert bool(jnp.all(jnp.isfinite(x)))
    np.testing.assert_allclose(Softplus().forward(x), y, rtol=1e-8)
