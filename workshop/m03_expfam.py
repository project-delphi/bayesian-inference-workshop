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
        raise NotImplementedError  # Module 3, Step 1

    def mean_params(self, eta: Array) -> Array:
        """Mean parameters mu(eta) = E_eta[t(x)] = grad A(eta), via jax.grad."""
        raise NotImplementedError  # Module 3, Step 4

    def fisher(self, eta: Array) -> Array:
        """Fisher information F(eta) = Cov_eta[t(x)] = Hessian of A at eta, via jax.hessian."""
        raise NotImplementedError  # Module 3, Step 5

    def score(self, x: Array, eta: Array) -> Array:
        """Score function grad_eta log p(x | eta) = t(x) - mean_params(eta)."""
        raise NotImplementedError  # Module 3, Step 5


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
        raise NotImplementedError  # Module 3, Step 1

    def log_base_measure(self, x):
        raise NotImplementedError  # Module 3, Step 1

    def log_partition(self, eta):
        """log(1 + sum_k exp eta_k)."""
        raise NotImplementedError  # Module 3, Step 1

    def sample(self, key, eta, n):
        logits = jnp.concatenate([eta, jnp.zeros(1)])
        return jax.random.categorical(key, logits, shape=(n,))

    def to_natural(self, p: Array) -> Array:
        """p is a length-K probability vector."""
        raise NotImplementedError  # Module 3, Step 1

    def to_standard(self, eta: Array) -> Array:
        """Return the length-K probability vector."""
        raise NotImplementedError  # Module 3, Step 1


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
        raise NotImplementedError  # Module 3, Step 2

    def log_base_measure(self, x):
        raise NotImplementedError  # Module 3, Step 2

    def log_partition(self, eta):
        """A(eta) = -1/4 eta1^T eta2^{-1} eta1 - 1/2 log det(-2 eta2) + d/2 log 2 pi."""
        raise NotImplementedError  # Module 3, Step 2

    def sample(self, key, eta, n):
        mu, cov = self.to_standard(eta)
        return jax.random.multivariate_normal(key, mu, cov, (n,))

    def to_natural(self, mu: Array, cov: Array) -> Array:
        raise NotImplementedError  # Module 3, Step 2

    def to_standard(self, eta: Array) -> tuple[Array, Array]:
        """Return (mu, cov)."""
        raise NotImplementedError  # Module 3, Step 2


class Gamma(ExpFam):
    """x > 0 with shape alpha and rate beta. eta = [alpha - 1, -beta], t(x) = [log x, x]."""

    dim = 2

    def sufficient_stats(self, x):
        raise NotImplementedError  # Module 3, Step 3

    def log_base_measure(self, x):
        raise NotImplementedError  # Module 3, Step 3

    def log_partition(self, eta):
        """A(eta) = lgamma(eta1 + 1) - (eta1 + 1) log(-eta2)."""
        raise NotImplementedError  # Module 3, Step 3

    def sample(self, key, eta, n):
        alpha, beta = self.to_standard(eta)
        return jax.random.gamma(key, alpha, (n,)) / beta

    def to_natural(self, alpha: Array, beta: Array) -> Array:
        raise NotImplementedError  # Module 3, Step 3

    def to_standard(self, eta: Array) -> tuple[Array, Array]:
        """Return (alpha, beta)."""
        raise NotImplementedError  # Module 3, Step 3


class Dirichlet(ExpFam):
    """x on the (K-1)-simplex with concentration alpha. eta = alpha - 1, t(x) = log x."""

    def __init__(self, n_categories: int):
        self.K = n_categories
        self.dim = n_categories

    def sufficient_stats(self, x):
        raise NotImplementedError  # Module 3, Step 3

    def log_base_measure(self, x):
        raise NotImplementedError  # Module 3, Step 3

    def log_partition(self, eta):
        """A(eta) = sum_k lgamma(eta_k + 1) - lgamma(sum_k (eta_k + 1))."""
        raise NotImplementedError  # Module 3, Step 3

    def sample(self, key, eta, n):
        return jax.random.dirichlet(key, self.to_standard(eta), (n,))

    def to_natural(self, alpha: Array) -> Array:
        raise NotImplementedError  # Module 3, Step 3

    def to_standard(self, eta: Array) -> Array:
        """Return alpha."""
        raise NotImplementedError  # Module 3, Step 3
