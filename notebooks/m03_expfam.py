# %% [markdown]
# # Module 3 — Exponential families: exploration
# The log-partition function is convex; its gradient is the mean map; its Hessian is the Fisher information.

# %%
import jax, jax.numpy as jnp, matplotlib.pyplot as plt
from notebooks._impl import impl
m03 = impl("m03_expfam")

# %%
fam = m03.Bernoulli()
etas = jnp.linspace(-6, 6, 200)
A = jax.vmap(lambda e: fam.log_partition(jnp.array([e])))(etas)
mu = jax.vmap(lambda e: fam.mean_params(jnp.array([e]))[0])(etas)
F = jax.vmap(lambda e: fam.fisher(jnp.array([e]))[0, 0])(etas)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ax[0].plot(etas, A); ax[0].set_title("A(eta) = softplus")
ax[1].plot(etas, mu); ax[1].set_title("grad A = sigmoid = E[x]")
ax[2].plot(etas, F); ax[2].set_title("Hessian A = p(1-p) = Var[x]")
plt.tight_layout(); plt.show()

# %% [markdown]
# Gamma family: the natural-parameter domain is eta1 > -1, eta2 < 0. Plot A over it.

# %%
g = m03.Gamma()
e1 = jnp.linspace(-0.9, 4, 100); e2 = jnp.linspace(-4, -0.1, 100)
E1, E2 = jnp.meshgrid(e1, e2)
AG = jax.vmap(jax.vmap(lambda a, b: g.log_partition(jnp.array([a, b]))))(E1, E2)
plt.contourf(E1, E2, AG, levels=40); plt.colorbar(); plt.xlabel("eta1 = alpha - 1"); plt.ylabel("eta2 = -beta")
plt.title("Gamma log-partition (convex)"); plt.show()

# %% [markdown]
# Monte Carlo check of the score identity E[score] = 0 and E[score score^T] = F for the Gamma.

# %%
eta = g.to_natural(jnp.asarray(3.5), jnp.asarray(1.7))
xs = g.sample(jax.random.key(0), eta, 100000)
S = jax.vmap(g.score, in_axes=(0, None))(xs, eta)
print("mean score:", S.mean(0))
print("E[s s^T]:\n", S.T @ S / S.shape[0])
print("Fisher:\n", g.fisher(eta))
