# %% [markdown]
# # Module 11 — Diagnostics: exploration
# The funnel of the hierarchical uplift model, centred vs non-centred, with divergent transitions marked.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m09, m11 = impl("m09_hmc"), impl("m11_diagnostics")

# %%
key = jax.random.key(3)
q0s = 0.1 * jax.random.normal(key, (4, m09.UPLIFT_DIM))
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for a, (name, lp, to_c) in zip(ax, [("centred", m09.uplift_centered_log_prob, lambda z: z), ("non-centred", m09.uplift_noncentered_log_prob, m09.uplift_noncentered_to_centered)]):
    ch, info, _ = m11.run_chains(lp, key, q0s, 1000, 1000)
    chc = to_c(ch)
    flat = chc.reshape(-1, chc.shape[-1]); div = np.asarray(~jnp.isfinite(info.energy_error) | (info.energy_error > 1000)).reshape(-1)
    a.scatter(flat[~div, 2], flat[~div, 1], s=3, alpha=0.3); a.scatter(flat[div, 2], flat[div, 1], s=12, color="r", label=f"divergent ({div.sum()})")
    a.set_xlabel("theta_1 (US-East uplift, pp)"); a.set_title(f"{name}: ESS(log tau) = {float(m11.ess(ch)[1]):.0f}"); a.legend()
ax[0].set_ylabel("log tau"); plt.tight_layout(); plt.show()

# %%
from workshop.data import saas_churn
X, y, _ = saas_churn(); X, y = jnp.asarray(X), jnp.asarray(y, float)
lj = lambda b: m11.churn_log_joint(b, X, y)
chains, info, _ = m11.run_chains(lj, key, jnp.zeros((4, 5)), 500, 1000)
s = m11.summarize(chains)
for k in ("mean", "sd", "rhat", "ess"): print(k, np.round(np.asarray(s[k]), 3))
flat = chains.reshape(-1, 5)
plt.imshow(np.corrcoef(np.asarray(flat).T), cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(); plt.title("churn posterior correlations"); plt.show()
