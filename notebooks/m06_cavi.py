# %% [markdown]
# # Module 6 — CAVI on the single-cell embedding

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m06, data = impl("m06_cavi"), impl("data")

# %%
x, z, means, w = data.single_cell_embedding()
x = jnp.asarray(x)
prior = m06.GMMPrior(1.0, 3.0, 0.35)
r0 = m06.init_responsibilities(jax.random.key(0), x, 4)
q, trace = m06.cavi(x, r0, prior, 60)
plt.plot(trace); plt.xlabel("sweep"); plt.ylabel("ELBO"); plt.title("CAVI: ELBO is non-decreasing"); plt.show()

# %%
pred = np.asarray(jnp.argmax(q.r, 1))
plt.scatter(x[:, 0], x[:, 1], c=pred, s=8, cmap="tab10")
plt.scatter(q.m[:, 0], q.m[:, 1], c="k", marker="x", s=120, label="q(mu) means")
plt.scatter(means[:, 0], means[:, 1], facecolors="none", edgecolors="r", s=160, label="true means")
plt.legend(); plt.title("Fitted components over cells"); plt.show()

# %%
ks = (2, 3, 4, 5, 6)
elbos = m06.elbo_by_k(jax.random.key(0), x, ks, prior, n_iter=80)
plt.plot(ks, elbos, "o-"); plt.xlabel("K"); plt.ylabel("final ELBO"); plt.title("ELBO as a model-selection score"); plt.show()

# %% [markdown]
# A random initialisation also ascends monotonically but can land in a worse optimum (label merging). Compare final ELBOs.

# %%
finals = []
for i in range(5):
    r_rand = jax.random.dirichlet(jax.random.key(100 + i), jnp.ones(4), (x.shape[0],))
    _, tr = m06.cavi(x, r_rand, prior, 60)
    finals.append(float(tr[-1]))
print("seeded:", float(trace[-1]), " random inits:", np.round(finals, 1))
