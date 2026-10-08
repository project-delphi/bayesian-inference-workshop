import jax
import jax.numpy as jnp
import numpy as np
import pytest

from workshop.data import torsion_angles
from workshop.m15b_diffusion import make_score_fn, reverse_drift, sample_reverse, train_score_model
from tests._util import key

CENTRES = np.array([[-1.1, -0.7], [-2.1, 2.3], [1.0, 0.8]])


def _assign(x):
    d = ((np.asarray(x)[:, None, :] - CENTRES[None]) ** 2).sum(-1)
    return np.bincount(d.argmin(1), minlength=3) / x.shape[0]


@pytest.fixture(scope="module")
def trained():
    X = jnp.asarray(torsion_angles())
    params, losses = train_score_model(key(1520), X, n_steps=6000, batch_size=256, lr=2e-3, hidden=128)
    return X, params, losses


def test_step4_reverse_drift_sign():
    # With the exact standard-normal score s = -x the reverse drift is +1/2 beta x
    from workshop.m15b_diffusion import beta

    x = jnp.array([1.0, -2.0])
    t = jnp.asarray(0.5)
    np.testing.assert_allclose(reverse_drift(lambda z, tt: -z, x, t), 0.5 * beta(t) * x, rtol=1e-12)


def test_step4_loss_decreases(trained):
    _, _, losses = trained
    assert bool(jnp.all(jnp.isfinite(losses)))
    # the DSM loss has an irreducible floor (noise variance), so compare to a fraction
    assert float(losses[-100:].mean()) < 0.7 * float(losses[:100].mean())


def test_step4_samples_recover_modes(trained):
    X, params, _ = trained
    xs = sample_reverse(key(1521), make_score_fn(params), n=2000, n_steps=400)
    assert xs.shape == (2000, 2) and bool(jnp.all(jnp.isfinite(xs)))
    p_data, p_gen = _assign(X), _assign(xs)
    assert np.abs(p_data - p_gen).max() < 0.12, (p_data, p_gen)
    np.testing.assert_allclose(xs.mean(0), X.mean(0), atol=0.2)
    np.testing.assert_allclose(xs.std(0), X.std(0), rtol=0.2)
