# %% [markdown]
# # Module 4 — KL as a Bregman divergence: exploration

# %%
import jax, jax.numpy as jnp, matplotlib.pyplot as plt
from notebooks._impl import impl
m03, m04 = impl("m03_expfam"), impl("m04_kl_bregman")

# %% [markdown]
# For the Bernoulli, draw A and its tangent at eta_p. The vertical gap at eta_q is KL(p || q).

# %%
fam = m03.Bernoulli()
etas = jnp.linspace(-5, 5, 300)
A = jax.vmap(lambda e: fam.log_partition(jnp.array([e])))(etas)
ep, eq = jnp.array([-1.0]), jnp.array([2.0])
tangent = fam.log_partition(ep) + fam.mean_params(ep)[0] * (etas - ep[0])
plt.plot(etas, A, label="A(eta)"); plt.plot(etas, tangent, "--", label="tangent at eta_p")
plt.vlines(eq[0], tangent[jnp.argmin(jnp.abs(etas - eq[0]))], fam.log_partition(eq), colors="r", label=f"KL = {float(m04.kl(fam, ep, eq)):.3f}")
plt.legend(); plt.title("Bregman divergence of the log-partition"); plt.show()

# %% [markdown]
# Monte Carlo KL vs closed form as a function of sample size.

# %%
g = m03.Gamma()
e1 = g.to_natural(jnp.asarray(3.0), jnp.asarray(2.0)); e2 = g.to_natural(jnp.asarray(1.5), jnp.asarray(0.7))
ns = [100, 1000, 10000, 100000]
ests = [m04.kl_monte_carlo(g, e1, e2, jax.random.key(i), n) for i, n in enumerate(ns)]
plt.errorbar(ns, [float(e) for e, _ in ests], yerr=[2 * float(s) for _, s in ests], fmt="o")
plt.axhline(float(m04.kl(g, e1, e2)), color="k", ls="--"); plt.xscale("log"); plt.title("MC KL with 2 SE bars vs closed form"); plt.show()

# %% [markdown]
# Newton inversion: convergence of mean_params(eta_k) to the target mu.

# %%
mu = g.mean_params(e1)
errs = []
for k in range(1, 12):
    eta_k = m04.natural_from_mean(g, mu, jnp.array([0.0, -1.0]), n_iter=k)
    errs.append(float(jnp.linalg.norm(g.mean_params(eta_k) - mu)))
plt.semilogy(range(1, 12), errs, "o-"); plt.xlabel("Newton iterations"); plt.ylabel("|grad A(eta_k) - mu|"); plt.show()
