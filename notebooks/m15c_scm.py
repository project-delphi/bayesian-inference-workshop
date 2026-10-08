# %% [markdown]
# # Track C — Structural causal models: exploration
# Observational versus interventional regression, the abducted posterior for one week, and the
# twin-network counterfactual probability as a function of the alternative spend.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt, scipy.stats
from notebooks._impl import impl
m15c = impl("m15c_scm"); data = impl("data")
scm = data.AdSpendSCM()
obs_data = {k: jnp.asarray(v) for k, v in data.ad_spend_observational().items()}

# %% [markdown]
# Observational cloud of (A, C) with its regression line, against the interventional mean line
# E[C | do(A = a)] = c_t b_a a. The gap in slope is the confounding bias through seasonality.

# %%
a, c = obs_data["A"], obs_data["C"]
grid = jnp.linspace(-4, 4, 50)
slope_obs = m15c.observational_slope(a, c)
inter = jnp.stack([m15c.interventional_samples(scm, {"A": g}, jax.random.key(i), 2000)["C"].mean() for i, g in enumerate(grid)])
plt.scatter(a, c, s=4, alpha=0.3, label="observational weeks")
plt.plot(grid, float(c.mean()) + float(slope_obs) * (grid - float(a.mean())), "r", label=f"regression slope {float(slope_obs):.2f}")
plt.plot(grid, inter, "k--", label=f"E[C | do(A)] slope {float(m15c.causal_effect(scm)):.2f}")
plt.xlabel("ad spend A"); plt.ylabel("conversions C"); plt.legend(); plt.title("Seeing versus doing"); plt.show()

# %% [markdown]
# One week with spend and traffic logged, conversions not yet attributed: the posterior over
# seasonality S and the counterfactual conversions under halved spend, HMC against the Gaussian closed form.

# %%
i = 3
obs = {"A": obs_data["A"][i], "T": obs_data["T"][i]}
a_new = obs["A"] / 2
post = m15c.counterfactual_posterior(scm, obs, {"A": a_new}, jax.random.key(0))
mean, sd = m15c.exact_counterfactual_C(scm, obs, a_new)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(post["S"], bins=40, density=True); ax[0].axvline(float(obs_data["S"][i]), color="k", ls="--", label="true S"); ax[0].set_title("abducted seasonality S | A, T"); ax[0].legend()
xs = np.linspace(float(mean) - 4 * float(sd), float(mean) + 4 * float(sd), 200)
ax[1].hist(post["C"], bins=40, density=True, label="HMC"); ax[1].plot(xs, scipy.stats.norm(float(mean), float(sd)).pdf(xs), "k", label="exact")
ax[1].axvline(float(obs_data["C"][i]), color="r", ls="--", label="actual C (unseen)"); ax[1].set_title("counterfactual C under do(A := A/2)"); ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# Twin network: probability that conversions would have fallen below a target, as a function of the
# alternative spend level.

# %%
target = 0.5
levels = jnp.linspace(-2.0, 2.0, 9)
probs = [float(m15c.probability_below(scm, obs, lv, target, jax.random.key(10 + k), n_warmup=200, n_samples=200)) for k, lv in enumerate(levels)]
exact = [scipy.stats.norm(*[float(v) for v in m15c.exact_counterfactual_C(scm, obs, lv)]).cdf(target) for lv in levels]
plt.plot(levels, probs, "o", label="twin-network HMC"); plt.plot(levels, exact, "k-", label="exact")
plt.axvline(float(obs["A"]), color="r", ls="--", label="actual spend"); plt.xlabel("counterfactual spend a'"); plt.ylabel(f"P(C_cf < {target} | A, T)")
plt.legend(); plt.title("Would conversions have missed the target?"); plt.show()
