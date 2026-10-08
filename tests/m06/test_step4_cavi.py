import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from workshop.data import single_cell_embedding
from workshop.m06_cavi import GMMPrior, cavi, init_responsibilities
from tests._util import key


def _fit():
    x, z, means, w = single_cell_embedding()
    x = jnp.asarray(x)
    prior = GMMPrior(1.0, 3.0, 0.35)
    r0 = init_responsibilities(key(230), x, 4)
    q, trace = cavi(x, r0, prior, 60)
    return x, z, means, w, q, trace


def test_step4_elbo_monotone():
    _, _, _, _, _, trace = _fit()
    assert trace.shape == (60,)
    assert bool(jnp.all(jnp.isfinite(trace)))
    assert bool(jnp.all(jnp.diff(trace) >= -1e-8))


def test_step4_recovers_means_and_labels():
    x, z, means, w, q, _ = _fit()
    cost = np.linalg.norm(np.asarray(q.m)[:, None, :] - means[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    assert cost[rows, cols].max() < 0.1
    pred = np.asarray(jnp.argmax(q.r, axis=1))
    perm = np.empty(4, dtype=int)
    perm[rows] = cols
    acc = np.mean(perm[pred] == z)
    assert acc > 0.95
    mix = np.asarray(q.alpha / q.alpha.sum())[rows]
    assert np.abs(mix - w[cols]).max() < 0.05


def test_step4_random_init_also_monotone():
    x, _, _, _ = single_cell_embedding()
    x = jnp.asarray(x)
    r0 = jax.random.dirichlet(key(231), jnp.ones(4), (600,))
    _, trace = cavi(x, r0, GMMPrior(1.0, 3.0, 0.35), 40)
    assert bool(jnp.all(jnp.diff(trace) >= -1e-8))
