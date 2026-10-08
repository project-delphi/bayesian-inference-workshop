import jax
import jax.numpy as jnp
import numpy as np

from workshop.m15b_diffusion import gaussian_mixture_score, perturbed_mixture_log_density
from tests._util import key

W = jnp.array([0.55, 0.35, 0.10])
M = jnp.array([[-1.1, -0.7], [-2.1, 2.3], [1.0, 0.8]])
C = jnp.stack([jnp.diag(jnp.array([0.25, 0.3]) ** 2), jnp.diag(jnp.array([0.35, 0.35]) ** 2), jnp.diag(jnp.array([0.2, 0.2]) ** 2)])


def test_step2_matches_autodiff_of_perturbed_density():
    xs = jax.random.normal(key(1501), (20, 2)) * 1.5
    for t in (0.01, 0.2, 0.6, 1.0):
        t = jnp.asarray(t)
        ref = jax.vmap(jax.grad(lambda x: perturbed_mixture_log_density(x, t, W, M, C)))(xs)
        out = jax.vmap(lambda x: gaussian_mixture_score(x, t, W, M, C))(xs)
        np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-10)


def test_step2_reduces_to_gaussian_score_for_one_component():
    w = jnp.array([1.0])
    m = jnp.array([[0.5, -0.5]])
    S = jnp.array([[[0.5, 0.1], [0.1, 0.3]]])
    x = jnp.array([1.0, 2.0])
    t = jnp.asarray(0.3)
    from workshop.m15b_diffusion import alpha_bar

    ab = alpha_bar(t)
    St = ab * S[0] + (1 - ab) * jnp.eye(2)
    ref = -jnp.linalg.solve(St, x - jnp.sqrt(ab) * m[0])
    np.testing.assert_allclose(gaussian_mixture_score(x, t, w, m, S), ref, rtol=1e-10)
