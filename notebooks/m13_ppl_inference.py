# %% [markdown]
# # Module 13 — Inference through the PPL: exploration
# Enzyme kinetics: SVI convergence, HMC vs VI posteriors, posterior predictive curve.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m13 = impl("m13_ppl_inference"); data = impl("data")
s, v, vmax_true, km_true, sd_true = data.enzyme_kinetics(); s, v = jnp.asarray(s), jnp.asarray(v)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for steps in (3000, 6000, 10000):
    params, elbo, spec = m13.svi(m13.enzyme_model, (s, v), jax.random.key(0), steps, lr=0.02, n_samples=32)
    ax[0].plot(np.convolve(np.asarray(elbo), np.ones(100) / 100, mode="valid"), label=f"{steps} steps")
ax[0].set_ylim(-80, -20); ax[0].legend(); ax[0].set_title("ELBO (smoothed): note the plateau")
ps = m13.posterior_samples_svi(params, spec, jax.random.key(1), 4000)
samples, infos, summary = m13.hmc(m13.enzyme_model, (s, v), jax.random.key(2), 400, 400)
ax[1].scatter(samples["Vmax"].reshape(-1), samples["Km"].reshape(-1), s=3, alpha=0.3, label="HMC")
ax[1].scatter(ps["Vmax"], ps["Km"], s=3, alpha=0.3, label="mean-field VI")
ax[1].plot(vmax_true, km_true, "k*", ms=14); ax[1].set_xlabel("Vmax"); ax[1].set_ylabel("Km"); ax[1].legend()
ax[1].set_title("Correlated posterior: VI under-disperses"); plt.tight_layout(); plt.show()
print({k: (round(float(summary[k][i]), 3)) for i, n in enumerate(summary["names"]) for k in ("rhat", "ess")} if False else dict(zip(summary["names"], np.round(np.asarray(summary["rhat"]), 3))))

# %%
flat = {k: x.reshape(-1) for k, x in samples.items()}
pred = m13.predictive(m13.enzyme_model, flat, jax.random.key(3), s, v)["v"]
grid = jnp.linspace(0.1, 35, 200)
curves = flat["Vmax"][:, None] * grid[None, :] / (flat["Km"][:, None] + grid[None, :])
lo, hi = np.percentile(np.asarray(curves), [5, 95], axis=0)
plt.fill_between(grid, lo, hi, alpha=0.3, label="90% band of mean curve"); plt.plot(grid, curves.mean(0), label="posterior mean curve")
plt.errorbar(s, pred.mean(0), yerr=2 * pred.std(0), fmt="none", ecolor="gray", label="predictive 2 SD")
plt.scatter(s, v, c="k", s=14, zorder=3, label="data"); plt.xscale("log"); plt.xlabel("substrate concentration"); plt.ylabel("velocity"); plt.legend(); plt.show()
