# %% [markdown]
# # Module 7 — Gradient estimator variance

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m07, data = impl("m07_gradients"), impl("data")

# %%
X, y, beta_true = (jnp.asarray(a, dtype=float) for a in data.saas_churn())
log_joint = lambda b: m07.churn_log_joint(b, X, y)
mean, cov = m07.laplace(log_joint, jnp.zeros(5))
sd = jnp.sqrt(jnp.diag(cov))
print("Laplace mean:", np.round(mean, 3)); print("Laplace sd:  ", np.round(sd, 3)); print("truth:       ", np.asarray(beta_true))

# %%
params = {"loc": mean + 0.3, "log_scale": jnp.log(sd) + 0.2}
ns = (4, 16, 64, 256, 1024)
study = m07.gradient_variance_study(jax.random.key(0), params, log_joint, ns, n_repeats=200)
for name, v in study.items():
    plt.loglog(ns, v, "o-", label=name)
plt.loglog(ns, study["score"][0] * (ns[0] / np.array(ns)), "k--", lw=0.8, label="1/n reference")
plt.xlabel("samples per estimate n"); plt.ylabel("total gradient variance"); plt.legend(); plt.title("ELBO gradient variance by estimator"); plt.show()

# %% [markdown]
# Per-coordinate view at n = 64: the control variate helps every coordinate; the pathwise estimator is orders of magnitude lower.

# %%
labels = [f"loc[{i}]" for i in range(5)] + [f"logsc[{i}]" for i in range(5)]
pos = np.arange(10)
for k, (name, est) in enumerate([("score", m07.elbo_grad_score), ("score_cv", m07.elbo_grad_score_cv), ("reparam", m07.elbo_grad_reparam)]):
    v = m07.estimator_variance(jax.random.key(1), est, params, log_joint, 64, 300)
    plt.bar(pos + 0.27 * k, v, width=0.27, label=name)
plt.yscale("log"); plt.xticks(pos + 0.27, labels, rotation=45); plt.legend(); plt.title("Per-coordinate variance, n = 64"); plt.show()
