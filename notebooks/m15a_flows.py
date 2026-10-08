# %% [markdown]
# # Module 15a — Normalising flows as variational families: exploration
# Mean-field Gaussian vs a RealNVP flow vs HMC on the funnel of the centred hierarchical uplift model.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m08, m09, F = impl("m08_bbvi"), impl("m09_hmc"), impl("m15a_flows")
lp, D = m09.uplift_centered_log_prob, m09.UPLIFT_DIM

# %%
ref = F.hmc_reference(jax.random.key(0))
mp, _ = m08.fit(lp, D, jax.random.key(1), 4000, lr=0.01, n_samples=16)
fp = F.init_flow(jax.random.key(2), D, 4, 32)
fp, trace = F.fit_flow_vi(lp, fp, jax.random.key(3), 12000, lr=2e-3, n_samples=64)
plt.plot(np.convolve(np.asarray(trace), np.ones(200) / 200, mode="valid")); plt.title("flow ELBO (running mean)"); plt.xlabel("step"); plt.show()

# %%
xf, lqf = F.flow_sample(jax.random.key(4), fp, 20000)
xm = m08.MeanFieldGaussian(D).sample(jax.random.key(5), mp, 20000)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
for a, (name, s) in zip(ax, [("HMC (non-centred, mapped)", ref), ("mean-field Gaussian", xm), ("RealNVP flow", xf)]):
    a.scatter(s[:4000, 2], s[:4000, 1], s=3, alpha=0.3); a.set_title(name); a.set_xlabel("theta_1 (US-East uplift, pp)")
ax[0].set_ylabel("log tau"); plt.tight_layout(); plt.show()

# %%
bins = np.linspace(-5, 3, 60)
for name, s in [("HMC", ref), ("mean-field", xm), ("flow", xf)]:
    plt.hist(np.asarray(s[:, 1]), bins=bins, density=True, histtype="step", lw=2, label=f"{name}: tail P(log tau < -1) = {float(F.tail_mass(s)):.2f}")
plt.legend(); plt.xlabel("log tau"); plt.title("Marginal of log tau"); plt.show()

# %%
ks = [1, 2, 4, 8, 16, 32, 64, 128]
bounds = [float(F.importance_weighted_elbo(lp, fp, jax.random.key(6), 1000, k)) for k in ks]
plt.semilogx(ks, bounds, "o-"); plt.xlabel("k"); plt.ylabel("IW bound L_k"); plt.title("Importance-weighted bound with the flow as proposal"); plt.show()
lw = jax.vmap(lp)(xf) - lqf
print("flow ESS fraction:", float(F.importance_ess(lw)) / lw.shape[0])
