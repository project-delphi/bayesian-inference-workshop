# %% [markdown]
# # Module 10 — Adaptation: exploration
# Step-size trace through the warm-up windows, adapted mass on a badly scaled target, and the U-turn study.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m09, m10 = impl("m09_hmc"), impl("m10_adaptation")

# %%
var = jnp.array([1.0, 100.0])
lp = lambda z: -0.5 * jnp.sum(z**2 / var)
is_slow, window_end = m10.warmup_schedule(1000)
# re-run warm-up manually to record the step size per iteration
key = jax.random.key(0)
q, da, wf, inv_mass = jnp.zeros(2), m10.dual_averaging_init(0.1), m10.welford_init(2), jnp.ones(2)
steps, masses = [], []
for i in range(1000):
    key, k_l, k_h = jax.random.split(key, 3)
    L = int(jax.random.randint(k_l, (), 1, 17))
    q, info = m09.hmc_step(lp, k_h, q, jnp.exp(da.log_step), L, inv_mass)
    da = m10.dual_averaging_update(da, info.accept_prob)
    if is_slow[i]:
        wf = m10.welford_update(wf, q)
    if window_end[i]:
        inv_mass = m10.welford_finalize(wf); wf = m10.welford_init(2); da = m10.dual_averaging_init(float(jnp.exp(da.log_step)))
    steps.append(float(jnp.exp(da.log_step))); masses.append(np.asarray(inv_mass))
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].semilogy(steps); ax[0].set_title("step size during warm-up")
for e in np.flatnonzero(window_end): ax[0].axvline(e, color="gray", ls=":")
ax[1].semilogy(np.array(masses)); ax[1].set_title("diagonal inverse mass (true variances 1 and 100)"); plt.tight_layout(); plt.show()

# %%
flags, first = m10.trajectory_length_study(m09.standard_normal_log_prob, jnp.array([1.0]), jnp.array([0.0]), 0.05, 200, jnp.ones(1))
plt.step(range(1, 201), flags); plt.axvline(np.pi / 0.05, color="r", ls="--", label="pi / eps"); plt.legend()
plt.title(f"U-turn criterion on the standard normal; first fires at L = {int(first)}"); plt.show()
