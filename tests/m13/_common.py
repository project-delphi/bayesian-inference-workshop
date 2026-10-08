import jax.numpy as jnp

from workshop.data import enzyme_kinetics, regional_uplift

S, V, VMAX_TRUE, KM_TRUE, SD_TRUE = enzyme_kinetics()
S, V = jnp.asarray(S), jnp.asarray(V)
Y, SIGMA = (jnp.asarray(a) for a in regional_uplift())
