import jax.numpy as jnp

from workshop.data import saas_churn
from workshop.m07_gradients import churn_log_joint

X, y, beta_true = (jnp.asarray(a, dtype=float) for a in saas_churn())
log_joint = lambda b: churn_log_joint(b, X, y)
