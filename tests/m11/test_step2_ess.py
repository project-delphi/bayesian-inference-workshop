import jax
import jax.numpy as jnp
import numpy as np

from workshop.m11_diagnostics import autocorrelation, ess
from tests._util import key


def test_step2_autocorrelation_matches_direct():
    x = jax.random.normal(key(280), (200,))
    rho = autocorrelation(x)
    xc = np.asarray(x) - np.asarray(x).mean()
    direct = np.array([np.dot(xc[: 200 - t], xc[t:]) for t in range(200)])
    direct /= direct[0]
    np.testing.assert_allclose(rho, direct, atol=1e-10)
    assert float(rho[0]) == 1.0


def test_step2_ess_iid():
    chains = jax.random.normal(key(281), (4, 2000, 2))
    e = ess(chains)
    assert e.shape == (2,)
    assert bool(jnp.all(jnp.abs(e / 8000 - 1.0) < 0.15)), e


def test_step2_ess_ar1():
    phi = 0.9
    m, n = 4, 5000
    eps = jax.random.normal(key(282), (m, n))

    def chain(e):
        return jax.lax.scan(lambda x, u: (phi * x + jnp.sqrt(1 - phi**2) * u, phi * x + jnp.sqrt(1 - phi**2) * u), 0.0, e)[1]

    chains = jax.vmap(chain)(eps)[:, 500:, None]
    e = ess(chains)
    expected = m * (n - 500) * (1 - phi) / (1 + phi)
    assert abs(float(e[0]) / expected - 1.0) < 0.25, (e, expected)


def test_step2_ess_antithetic_exceeds_n():
    # a strongly negatively autocorrelated chain has ESS above the number of draws
    x = jax.random.normal(key(283), (1, 4001, 1))
    anti = x[:, 1:] - 0.9 * x[:, :-1]  # MA(1) with negative lag-1 autocorrelation
    assert float(ess(anti)[0]) > 4000
