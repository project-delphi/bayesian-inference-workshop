"""Module 3 — Exponential-family core.

A family of densities p(x | eta) = h(x) exp( eta . t(x) - A(eta) ) is specified by its
sufficient statistic t, log base measure log h and log-partition A. Everything else in
this module (log density, mean parameters, Fisher information, score) is derived from
those three functions by autodiff. Each concrete family also supplies `sample` and
conversions between its textbook parameters and natural parameters.

Conventions
-----------
- Natural parameters `eta` are always a flat 1-D array of length `self.dim`.
- `sufficient_stats(x)` returns a 1-D array of length `self.dim` for one observation x.
- Functions act on a single observation; use jax.vmap for batches.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

Array = jax.Array


class ExpFam:
    """Base class. Subclasses define t, log h, A, sample and parameter conversions."""

    dim: int

    # --- to be implemented by each family -------------------------------------------
    def sufficient_stats(self, x: Array) -> Array:
        raise NotImplementedError

    def log_base_measure(self, x: Array) -> Array:
        raise NotImplementedError

    def log_partition(self, eta: Array) -> Array:
        raise NotImplementedError

    def sample(self, key: Array, eta: Array, n: int) -> Array:
        raise NotImplementedError

    # --- derived quantities, shared by every family ---------------------------------
    def log_prob(self, x: Array, eta: Array) -> Array:
        """log p(x | eta) = log h(x) + eta . t(x) - A(eta) for a single observation x."""
        # [m03 step 1]
        return self.log_base_measure(x) + jnp.dot(eta, self.sufficient_stats(x)) - self.log_partition(eta)

    def mean_params(self, eta: Array) -> Array:
        """Mean parameters mu(eta) = E_eta[t(x)] = grad A(eta), via jax.grad."""
        # [m03 step 4]
        return jax.grad(self.log_partition)(eta)

    def fisher(self, eta: Array) -> Array:
        """Fisher information F(eta) = Cov_eta[t(x)] = Hessian of A at eta, via jax.hessian."""
        # [m03 step 5]
        return jax.hessian(self.log_partition)(eta)

    def score(self, x: Array, eta: Array) -> Array:
        """Score function grad_eta log p(x | eta) = t(x) - mean_params(eta)."""
        # [m03 step 5]
        return self.sufficient_stats(x) - self.mean_params(eta)


class Bernoulli(ExpFam):
    """x in {0, 1}. eta = [logit p]. Fully implemented as the worked example."""

    dim = 1

    def sufficient_stats(self, x):
        return jnp.reshape(x, (1,)).astype(float)

    def log_base_measure(self, x):
        return jnp.zeros(())

    def log_partition(self, eta):
        return jax.nn.softplus(eta[0])

    def sample(self, key, eta, n):
        return jax.random.bernoulli(key, jax.nn.sigmoid(eta[0]), (n,)).astype(float)

    def to_natural(self, p: Array) -> Array:
        return jnp.reshape(jnp.log(p) - jnp.log1p(-p), (1,))

    def to_standard(self, eta: Array) -> Array:
        return jax.nn.sigmoid(eta[0])


class Categorical(ExpFam):
    """x in {0, ..., K-1}. Minimal representation with K-1 natural parameters:
    eta_k = log(p_k / p_{K-1}) for k < K-1, so the last category is the reference."""

    def __init__(self, n_categories: int):
        self.K = n_categories
        self.dim = n_categories - 1

    def sufficient_stats(self, x):
        """First K-1 entries of the one-hot encoding of x."""
        # [m03 step 1]
        return jax.nn.one_hot(x, self.K)[: self.K - 1]

    def log_base_measure(self, x):
        # [m03 step 1]
        return jnp.zeros(())

    def log_partition(self, eta):
        """log(1 + sum_k exp eta_k)."""
        # [m03 step 1]
        return jax.nn.logsumexp(jnp.concatenate([eta, jnp.zeros(1)]))

    def sample(self, key, eta, n):
        logits = jnp.concatenate([eta, jnp.zeros(1)])
        return jax.random.categorical(key, logits, shape=(n,))

    def to_natural(self, p: Array) -> Array:
        """p is a length-K probability vector."""
        # [m03 step 1]
        return jnp.log(p[:-1]) - jnp.log(p[-1])

    def to_standard(self, eta: Array) -> Array:
        """Return the length-K probability vector."""
        # [m03 step 1]
        return jax.nn.softmax(jnp.concatenate([eta, jnp.zeros(1)]))


class Gaussian(ExpFam):
    """x in R^d with full covariance.

    eta = concat( Lambda mu , vec(-Lambda / 2) )  where Lambda = Sigma^{-1},
    t(x) = concat( x , vec(x x^T) ), log h(x) = 0 (the -d/2 log 2 pi is inside A).
    The vec(.) block is over-complete (x x^T is symmetric) so the Fisher matrix of
    this representation is singular; the mean parameters are still correct.
    """

    def __init__(self, d: int):
        self.d = d
        self.dim = d + d * d

    def _split(self, eta):
        d = self.d
        return eta[:d], eta[d:].reshape(d, d)

    def sufficient_stats(self, x):
        # [m03 step 2]
        return jnp.concatenate([x, jnp.outer(x, x).ravel()])

    def log_base_measure(self, x):
        # [m03 step 2]
        return jnp.zeros(())

    def log_partition(self, eta):
        """A(eta) = -1/4 eta1^T eta2^{-1} eta1 - 1/2 log det(-2 eta2) + d/2 log 2 pi."""
        # [m03 step 2]
        eta1, eta2 = self._split(eta)
        Lam = -2.0 * eta2
        _, logdet = jnp.linalg.slogdet(Lam)
        # eta2^{-1} = -2 Lam^{-1}, so -1/4 eta1^T eta2^{-1} eta1 = 1/2 eta1^T Lam^{-1} eta1.
        quad = eta1 @ jnp.linalg.solve(Lam, eta1)
        return 0.5 * quad - 0.5 * logdet + 0.5 * self.d * jnp.log(2 * jnp.pi)

    def sample(self, key, eta, n):
        mu, cov = self.to_standard(eta)
        return jax.random.multivariate_normal(key, mu, cov, (n,))

    def to_natural(self, mu: Array, cov: Array) -> Array:
        # [m03 step 2]
        Lam = jnp.linalg.inv(cov)
        return jnp.concatenate([Lam @ mu, (-0.5 * Lam).ravel()])

    def to_standard(self, eta: Array) -> tuple[Array, Array]:
        """Return (mu, cov)."""
        # [m03 step 2]
        eta1, eta2 = self._split(eta)
        Lam = -2.0 * eta2
        cov = jnp.linalg.inv(Lam)
        return cov @ eta1, cov


class Gamma(ExpFam):
    """x > 0 with shape alpha and rate beta. eta = [alpha - 1, -beta], t(x) = [log x, x]."""

    dim = 2

    def sufficient_stats(self, x):
        # [m03 step 3]
        return jnp.stack([jnp.log(x), x])

    def log_base_measure(self, x):
        # [m03 step 3]
        return jnp.zeros(())

    def log_partition(self, eta):
        """A(eta) = lgamma(eta1 + 1) - (eta1 + 1) log(-eta2)."""
        # [m03 step 3]
        alpha = eta[0] + 1.0
        beta = -eta[1]
        return gammaln(alpha) - alpha * jnp.log(beta)

    def sample(self, key, eta, n):
        alpha, beta = self.to_standard(eta)
        return jax.random.gamma(key, alpha, (n,)) / beta

    def to_natural(self, alpha: Array, beta: Array) -> Array:
        # [m03 step 3]
        return jnp.stack([alpha - 1.0, -beta])

    def to_standard(self, eta: Array) -> tuple[Array, Array]:
        """Return (alpha, beta)."""
        # [m03 step 3]
        return eta[0] + 1.0, -eta[1]


class Dirichlet(ExpFam):
    """x on the (K-1)-simplex with concentration alpha. eta = alpha - 1, t(x) = log x."""

    def __init__(self, n_categories: int):
        self.K = n_categories
        self.dim = n_categories

    def sufficient_stats(self, x):
        # [m03 step 3]
        return jnp.log(x)

    def log_base_measure(self, x):
        # [m03 step 3]
        return jnp.zeros(())

    def log_partition(self, eta):
        """A(eta) = sum_k lgamma(eta_k + 1) - lgamma(sum_k (eta_k + 1))."""
        # [m03 step 3]
        alpha = eta + 1.0
        return jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))

    def sample(self, key, eta, n):
        return jax.random.dirichlet(key, self.to_standard(eta), (n,))

    def to_natural(self, alpha: Array) -> Array:
        # [m03 step 3]
        return alpha - 1.0

    def to_standard(self, eta: Array) -> Array:
        """Return alpha."""
        # [m03 step 3]
        return eta + 1.0
