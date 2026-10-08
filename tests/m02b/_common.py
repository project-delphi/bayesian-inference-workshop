import jax.numpy as jnp
import numpy as np

from workshop.data import qpcr_log_expression

X = jnp.asarray(qpcr_log_expression()[0])
MU_GRID = jnp.linspace(1.2, 2.4, 241)
LOGLAM_GRID = jnp.linspace(0.0, 4.5, 301)


def np_log_joint_constrained(mu, lam, x=np.asarray(X)):
    """Reference: log p(x, mu, lam) over the constrained space, numpy only."""
    from scipy.special import gammaln

    lp_mu = -0.5 * (mu / 10.0) ** 2 - np.log(10.0) - 0.5 * np.log(2 * np.pi)
    lp_lam = 2.0 * np.log(0.5) - gammaln(2.0) + np.log(lam) - 0.5 * lam
    ll = 0.5 * len(x) * (np.log(lam) - np.log(2 * np.pi)) - 0.5 * lam * np.sum((x - mu) ** 2)
    return lp_mu + lp_lam + ll
