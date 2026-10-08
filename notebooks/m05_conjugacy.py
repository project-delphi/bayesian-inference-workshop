# %% [markdown]
# # Module 5 — Conjugacy: exploration
# Posterior evolution on the A/B test, channel attribution posterior, and the qPCR Normal-Gamma posterior.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt, scipy.stats
from notebooks._impl import impl
m05 = impl("m05_conjugacy"); data = impl("data")

# %%
ab = data.ab_test_conversions()
prior = m05.BetaBernoulli.from_beta(1.0, 1.0)
_, chis, nus = m05.sequential_update(prior, jnp.asarray(ab.a, float))
a = chis[:, 0]; b = nus - a
plt.plot(a / (a + b), label="posterior mean"); plt.axhline(ab.true_rate_a, color="k", ls="--", label="truth")
plt.fill_between(range(len(a)), scipy.stats.beta(a, b).ppf(0.05), scipy.stats.beta(a, b).ppf(0.95), alpha=0.3)
plt.title("Variant A: streaming Beta posterior"); plt.legend(); plt.show()

# %%
pa = m05.posterior_update(prior, jnp.asarray(ab.a, float)); pb = m05.posterior_update(prior, jnp.asarray(ab.b, float))
print("P(B beats A) =", float(m05.prob_b_beats_a(pa, pb, jax.random.key(0), 200000)))

# %%
labels, p_true = data.channel_attribution()
post = m05.posterior_update(m05.DirichletCategorical.from_alpha(jnp.ones(6) * 0.5), jnp.asarray(labels))
alpha = np.asarray(post.alpha()); mean = alpha / alpha.sum(); sd = np.sqrt(mean * (1 - mean) / (alpha.sum() + 1))
pos = np.arange(6)
plt.bar(pos, mean, yerr=2 * sd, capsize=3); plt.plot(pos, p_true, "k_", ms=20, label="truth")
plt.xticks(pos, data.CHANNELS, rotation=30); plt.title("Channel attribution: posterior mean, 2 SD"); plt.legend(); plt.show()

# %%
xs, mu_true, lam_true = data.qpcr_log_expression()
ng = m05.normal_gamma_update(m05.NormalGamma(jnp.asarray(0.0), jnp.asarray(0.1), jnp.asarray(2.0), jnp.asarray(0.5)), jnp.asarray(xs))
mus = np.linspace(1.0, 2.6, 300); lams = np.linspace(1, 30, 300); M, L = np.meshgrid(mus, lams)
logp = scipy.stats.norm(float(ng.m), 1 / np.sqrt(float(ng.kappa) * L)).logpdf(M) + scipy.stats.gamma(float(ng.alpha), scale=1 / float(ng.beta)).logpdf(L)
plt.contourf(M, L, np.exp(logp), levels=30); plt.plot(mu_true, lam_true, "r*", ms=14)
plt.xlabel("mu (log2 fold change)"); plt.ylabel("lambda (precision)"); plt.title("qPCR Normal-Gamma posterior; star = truth"); plt.show()
