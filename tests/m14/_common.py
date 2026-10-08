import jax.numpy as jnp

from workshop.data import gene_programs

X_NP, Z_NP, MASKS = gene_programs()
X = jnp.asarray(X_NP)
Z = jnp.asarray(Z_NP)
L = 4
