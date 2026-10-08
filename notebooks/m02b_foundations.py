# %% [markdown]
# # Module 2b — Bayesian inference from first principles: exploration
# One posterior, four routes: grid, Gaussian by VI, Laplace, random-walk Metropolis.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m = impl("m02b_foundations"); data = impl("data")
x = jnp.asarray(data.qpcr_log_expression()[0])
mu_grid = jnp.linspace(1.2, 2.4, 241); ll_grid = jnp.linspace(0.0, 4.5, 301)
log_post, log_ev = m.grid_log_posterior(m.log_joint, x, mu_grid, ll_grid)
mean, cov = m.grid_moments(log_post, mu_grid, ll_grid)
print("log evidence", float(log_ev), "posterior mean", mean, "sd", np.sqrt(np.diag(cov)))

# %% [markdown]
# The exact (gridded) posterior over (mu, log lambda). Note the shape: nearly Gaussian in mu, skewed in log lambda.

# %%
M, L = np.meshgrid(np.asarray(mu_grid), np.asarray(ll_grid), indexing="ij")
plt.contourf(M, L, np.exp(np.asarray(log_post)), levels=30); plt.colorbar(label="posterior density")
plt.xlabel("mu"); plt.ylabel("log lambda"); plt.title("Grid posterior, qPCR replicates"); plt.show()

# %% [markdown]
# Variational inference in miniature: a diagonal Gaussian fitted by maximising the ELBO. The ELBO never exceeds the log evidence.

# %%
eps = jax.random.normal(jax.random.key(0), (256, 2))
q_mean, q_log_sd, elbo = m.fit_gaussian_vi(m.log_joint, x, eps, jnp.array([1.5, 1.0]), jnp.log(jnp.array([0.3, 0.6])))
plt.plot(elbo, label="ELBO"); plt.axhline(float(log_ev), color="k", ls="--", label="log evidence (grid)")
plt.ylim(float(log_ev) - 5, float(log_ev) + 0.5); plt.legend(); plt.xlabel("step"); plt.title("ELBO is a lower bound"); plt.show()
print("VI mean", q_mean, "VI sd", np.exp(q_log_sd), "grid sd", np.sqrt(np.diag(cov)))

# %% [markdown]
# Laplace: a Gaussian at the mode. Compare the three Gaussians against the grid marginal of log lambda.

# %%
mode, lcov = m.laplace_approx(m.log_joint, x, jnp.array([1.5, 1.0]))
d_mu = float(mu_grid[1] - mu_grid[0])
marg = np.exp(np.asarray(log_post)).sum(0) * d_mu
ll = np.asarray(ll_grid)
plt.plot(ll, marg, "k", label="grid marginal")
for lab, mu_, sd_ in [("VI", float(q_mean[1]), float(np.exp(q_log_sd[1]))), ("Laplace", float(mode[1]), float(np.sqrt(lcov[1, 1])))]:
    plt.plot(ll, np.exp(-0.5 * ((ll - mu_) / sd_) ** 2) / (sd_ * np.sqrt(2 * np.pi)), label=lab)
plt.xlabel("log lambda"); plt.legend(); plt.title("Marginal of log precision: exact vs two Gaussians"); plt.show()

# %% [markdown]
# MCMC in miniature: random-walk Metropolis. Trace plot and the sample histogram against the grid marginal.

# %%
samples, acc = m.random_walk_metropolis(lambda t: m.log_joint(t, x), jax.random.key(1), jnp.array([1.5, 1.0]), 20000, 0.35)
print("acceptance", float(acc))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(samples[:2000, 1], lw=0.5); ax[0].set_title("trace of log lambda (first 2000)")
ax[1].hist(np.asarray(samples[2000:, 1]), bins=60, density=True, alpha=0.5, label="MH samples"); ax[1].plot(ll, marg, "k", label="grid"); ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# Why grids stop at a handful of parameters.

# %%
for d in [1, 2, 3, 5, 10, 20]:
    print(f"d={d:2d}: {m.grid_cost(d, 200):.3g} evaluations")
