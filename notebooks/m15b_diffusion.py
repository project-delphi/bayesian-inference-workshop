# %% [markdown]
# # Track B — Score-based diffusion: exploration
# Forward noising of torsion angles, training curve, reverse-SDE samples, probability-flow trajectories, and likelihood.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m15b, data = impl("m15b_diffusion"), impl("data")
X = jnp.asarray(data.torsion_angles())
key = jax.random.key(0)

# %% [markdown]
# Forward process: the data at t = 0, 0.2, 0.5, 1.

# %%
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for a, t in zip(ax, [0.0, 0.2, 0.5, 1.0]):
    xt = jax.vmap(lambda x, k: m15b.sample_forward(k, x, jnp.asarray(t)))(X, jax.random.split(key, X.shape[0]))
    a.scatter(xt[:, 0], xt[:, 1], s=2); a.set_title(f"t = {t}"); a.set_xlim(-4, 4); a.set_ylim(-4, 4)
plt.tight_layout(); plt.show()

# %% [markdown]
# Train the score network and look at the loss curve (note the irreducible floor).

# %%
params, losses = m15b.train_score_model(jax.random.key(1), X, n_steps=6000, batch_size=256, lr=2e-3)
plt.plot(np.convolve(np.asarray(losses), np.ones(50) / 50, mode="valid")); plt.xlabel("step"); plt.ylabel("DSM loss (50-step average)"); plt.show()
score_fn = m15b.make_score_fn(params)

# %% [markdown]
# Reverse-SDE samples for increasing numbers of Euler-Maruyama steps.

# %%
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
ax[0].scatter(X[:2000, 0], X[:2000, 1], s=2); ax[0].set_title("data")
for a, n in zip(ax[1:], [50, 200, 1000]):
    xs = m15b.sample_reverse(jax.random.key(2), score_fn, 2000, n)
    a.scatter(xs[:, 0], xs[:, 1], s=2); a.set_title(f"{n} reverse steps")
for a in ax: a.set_xlim(-4, 4); a.set_ylim(-4, 4)
plt.tight_layout(); plt.show()

# %% [markdown]
# Learned score field at t = 0.1 and probability-flow ODE trajectories from noise to data.

# %%
g = jnp.linspace(-3.5, 3.5, 20)
G1, G2 = jnp.meshgrid(g, g)
pts = jnp.stack([G1.ravel(), G2.ravel()], 1)
S = jax.vmap(lambda p: score_fn(p, jnp.asarray(0.1)))(pts)
plt.quiver(pts[:, 0], pts[:, 1], S[:, 0], S[:, 1], angles="xy", scale=300)
plt.title("learned score at t = 0.1"); plt.show()

# %%
def flow_trajectory(x1, n_steps=200):
    dt = (1.0 - m15b.T_EPS) / n_steps
    def step(x, t):
        x_new = x - dt * m15b.probability_flow_drift(score_fn, x, t)
        return x_new, x_new
    _, traj = jax.lax.scan(step, x1, 1.0 - dt * jnp.arange(n_steps))
    return traj
starts = jax.random.normal(jax.random.key(3), (40, 2))
trajs = jax.vmap(flow_trajectory)(starts)
plt.scatter(X[:2000, 0], X[:2000, 1], s=2, alpha=0.3)
for tr in trajs: plt.plot(tr[:, 0], tr[:, 1], lw=0.7, color="k")
plt.title("probability-flow ODE paths, noise to data"); plt.xlim(-4, 4); plt.ylim(-4, 4); plt.show()

# %% [markdown]
# Held-out log-likelihood: exact divergence vs Hutchinson, and against a single Gaussian.

# %%
import scipy.stats
X_test = jnp.asarray(data.torsion_angles(seed=1, n=200))
ll_exact = jax.vmap(lambda x: m15b.ode_log_likelihood_exact(score_fn, x, 200))(X_test)
ll_hutch = jax.vmap(lambda x, k: m15b.ode_log_likelihood(score_fn, x, k, 200))(X_test, jax.random.split(jax.random.key(4), 200))
gauss = scipy.stats.multivariate_normal(np.asarray(X.mean(0)), np.cov(np.asarray(X).T)).logpdf(np.asarray(X_test))
print("mean held-out log-lik: exact", float(ll_exact.mean()), "hutchinson", float(ll_hutch.mean()), "single gaussian", gauss.mean())
plt.scatter(ll_exact, ll_hutch, s=8); plt.plot([-6, 1], [-6, 1], "k--"); plt.xlabel("exact divergence"); plt.ylabel("Hutchinson"); plt.title("per-point ODE log-likelihood"); plt.show()
