import jax
import jax.numpy as jnp
import numpy as np

from workshop.m14_vae import elbo_analytic_kl, independent_bernoulli_baseline, train_vae
from tests.m14._common import L, X
from tests._util import key


def test_step4_training_beats_independent_baseline():
    params, trace_ = train_vae(key(110), X, latent_dim=L, n_steps=1500, batch_size=128, lr=1e-3)
    assert trace_.shape == (1500,)
    assert float(trace_[-100:].mean()) > float(trace_[:100].mean())
    full = jnp.mean(jnp.stack([elbo_analytic_kl(k, params, X, L) for k in jax.random.split(key(111), 10)]))
    baseline = independent_bernoulli_baseline(X)
    assert float(full) > float(baseline) + 2.0, (float(full), float(baseline))


def test_step4_baseline_value():
    # 64 genes, column means as fitted probabilities
    p = np.clip(np.asarray(X).mean(0), 1e-6, 1 - 1e-6)
    ref = np.sum(p * np.log(p) + (1 - p) * np.log1p(-p))
    np.testing.assert_allclose(independent_bernoulli_baseline(X), ref, rtol=1e-10)
    assert -45 < ref < -35
