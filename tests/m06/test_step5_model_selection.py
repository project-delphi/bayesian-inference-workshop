import jax.numpy as jnp
import numpy as np

from workshop.data import single_cell_embedding
from workshop.m06_cavi import GMMPrior, elbo_by_k
from tests._util import key


def test_step5_elbo_prefers_four_components():
    x, _, _, _ = single_cell_embedding()
    x = jnp.asarray(x)
    ks = (2, 3, 4, 5, 6)
    elbos = elbo_by_k(key(240), x, ks, GMMPrior(1.0, 3.0, 0.35), n_iter=80)
    assert elbos.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(elbos)))
    assert ks[int(jnp.argmax(elbos))] == 4
    # Beyond the true K the gain is small compared with the gain up to K=4.
    assert elbos[2] - elbos[1] > 10 * abs(elbos[4] - elbos[2])
