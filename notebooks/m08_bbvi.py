# %% [markdown]
# # Module 8 — Black-box VI engine: fits and diagnostics

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m07, m08, data = impl("m07_gradients"), impl("m08_bbvi"), impl("data")

# %%
X, y, _ = (jnp.asarray(a, dtype=float) for a in data.saas_churn())
lj = lambda b: m07.churn_log_joint(b, X, y)
mean, cov = m07.laplace(lj, jnp.zeros(5)); sd = jnp.sqrt(jnp.diag(cov))
params, trace = m08.fit(lj, 5, jax.random.key(0), n_steps=2500, lr=0.02, n_samples=8)
ev = m07.laplace_log_evidence(lj, mean, cov)
plt.plot(trace, lw=0.6); plt.axhline(ev, color="k", ls="--", label="Laplace evidence"); plt.ylim(float(ev) - 30, float(ev) + 5)
plt.xlabel("step"); plt.ylabel("ELBO estimate"); plt.legend(); plt.title("Churn posterior: ELBO trace"); plt.show()

# %%
names = data.CHURN_FEATURES
pos = np.arange(5)
plt.errorbar(pos - 0.1, mean, yerr=2 * sd, fmt="o", label="Laplace (2 SD)")
plt.errorbar(pos + 0.1, params["loc"], yerr=2 * jnp.exp(params["log_scale"]), fmt="s", label="mean-field VI (2 SD)")
plt.xticks(pos, names, rotation=30); plt.legend(); plt.title("Posterior marginals"); plt.show()

# %% [markdown]
# Stochastic VI on 50k ad impressions: minibatches of 500, compared with a full-data Laplace fit.

# %%
Xa, ya, beta_a = (jnp.asarray(a, dtype=float) for a in data.ad_clicks())
log_prior = lambda b: -0.5 * jnp.sum(b**2) / 2.5**2
log_lik = lambda b, Xb, yb: jnp.sum(yb * jax.nn.log_sigmoid(Xb @ b) + (1 - yb) * jax.nn.log_sigmoid(-(Xb @ b)))
mean_a, cov_a = m07.laplace(lambda b: log_prior(b) + log_lik(b, Xa, ya), jnp.zeros(9))
p_svi, tr_svi = m08.fit_svi(log_prior, log_lik, (Xa, ya), 9, jax.random.key(1), n_steps=3000, batch_size=500, lr=0.02, n_samples=4)
plt.plot(np.convolve(np.asarray(tr_svi), np.ones(50) / 50, mode="valid"), lw=0.8); plt.xlabel("step"); plt.ylabel("minibatch ELBO (50-step average)"); plt.title("SVI trace"); plt.show()
plt.errorbar(np.arange(9) - 0.1, mean_a, yerr=2 * jnp.sqrt(jnp.diag(cov_a)), fmt="o", label="full-data Laplace")
plt.errorbar(np.arange(9) + 0.1, p_svi["loc"], yerr=2 * jnp.exp(p_svi["log_scale"]), fmt="s", label="SVI (B = 500)")
plt.plot(np.arange(9), beta_a, "k_", ms=14, label="truth"); plt.legend(); plt.title("Ad CTR coefficients"); plt.show()

# %% [markdown]
# Bijectors: a Gamma-distributed parameter on the unconstrained line.

# %%
b = m08.Softplus()
zeta = jnp.linspace(-4, 4, 200)
alpha_, beta_ = 3.0, 1.5
log_p_theta = lambda t: (alpha_ - 1) * jnp.log(t) - beta_ * t + alpha_ * jnp.log(beta_) - jax.scipy.special.gammaln(alpha_)
log_p_zeta = jax.vmap(lambda z: log_p_theta(b.forward(z)) + b.log_det_jacobian(z))(zeta)
plt.plot(zeta, jnp.exp(log_p_zeta)); plt.title("Gamma(3, 1.5) pushed to R through softplus^{-1}"); plt.xlabel("zeta"); plt.show()
