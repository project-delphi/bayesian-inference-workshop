import jax
import jax.numpy as jnp
import numpy as np

from workshop.m14_vae import latent_means, program_separation, reconstruct, train_vae
from tests.m14._common import L, X, Z
from tests._util import key


def test_step5_latent_space_groups_cells_by_program():
    params, _ = train_vae(key(120), X, latent_dim=L, n_steps=1500)
    lat = latent_means(params, X, L)
    assert lat.shape == (X.shape[0], L)
    sep = program_separation(lat[:600], Z[:600])
    assert sep < 0.6, sep
    # random labels give no separation
    perm = jax.random.permutation(key(121), Z[:600])
    assert program_separation(lat[:600], perm) > 0.9


def test_step5_reconstruction_is_probabilistic_and_accurate():
    params, _ = train_vae(key(122), X, latent_dim=L, n_steps=1500)
    xb = X[:200]
    r = reconstruct(params, xb, key(123), L)
    assert r.shape == xb.shape and bool(jnp.all((r >= 0) & (r <= 1)))
    acc = float(((r > 0.5) == xb).mean())
    assert acc > 0.8, acc
    r2 = reconstruct(params, xb, key(124), L)
    assert not np.allclose(r, r2)
