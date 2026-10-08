# %% [markdown]
# # Module 12 — Effect handlers: exploration
# Inspect traces, see what each handler does to a message, and check the log density against Module 9.

# %%
import jax, jax.numpy as jnp, matplotlib.pyplot as plt
from notebooks._impl import impl
h = impl("m12_handlers"); data = impl("data"); m09 = impl("m09_hmc")
Y, SIGMA = (jnp.asarray(a) for a in data.regional_uplift())

# %%
tr = h.trace(h.seed(h.uplift_model, jax.random.key(0))).get_trace(Y, SIGMA)
for name, site in tr.items():
    print(f"{name:4s} {site['type']:6s} observed={site['is_observed']!s:5s} shape={jnp.shape(site['value'])}")

# %% [markdown]
# Prior predictive draws through the trace: 200 executions, collect theta = mu + tau * eta.

# %%
def prior_theta(k):
    t = h.trace(h.seed(h.uplift_model, k)).get_trace(Y, SIGMA)
    return t["mu"]["value"] + t["tau"]["value"] * t["eta"]["value"]
thetas = jnp.stack([prior_theta(k) for k in jax.random.split(jax.random.key(1), 200)])
plt.boxplot(thetas, tick_labels=data.REGIONS); plt.plot(range(1, 9), Y, "r_", ms=18, label="observed uplift")
plt.legend(); plt.title("Prior predictive theta_j (boxes) vs observed y_j"); plt.xticks(rotation=30); plt.show()

# %% [markdown]
# The log density as a function of log tau, with the other latents fixed, compared with Module 9's hand-written target.

# %%
z0 = jax.random.normal(jax.random.key(2), (10,))
lts = jnp.linspace(-3, 2, 100)
def ppl(lt):
    z = z0.at[1].set(lt)
    return h.log_density(h.uplift_model, {"mu": z[0], "tau": jnp.exp(z[1]), "eta": z[2:]}, Y, SIGMA)[0] + lt
ref = jax.vmap(lambda lt: m09.uplift_noncentered_log_prob(z0.at[1].set(lt)))(lts)
ours = jax.vmap(ppl)(lts)
plt.plot(lts, ours - ours[0], label="PPL log_density + log tau"); plt.plot(lts, ref - ref[0], "--", label="Module 9 target")
plt.xlabel("log tau"); plt.legend(); plt.title("Same function up to a constant"); plt.show()
