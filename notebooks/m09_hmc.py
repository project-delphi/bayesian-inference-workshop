# %% [markdown]
# # Module 9 — Hamiltonian dynamics: exploration
# Leapfrog trajectories on the correlated Gaussian, energy error against step size, and RWMH vs HMC traces.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m09 = impl("m09_hmc")

# %%
grad = jax.grad(m09.gaussian_2d_log_prob)
q0, p0 = jnp.array([0.5, -1.0]), jnp.array([1.0, 0.3])
for eps, c in [(0.5, "C0"), (0.1, "C1")]:
    qs = [q0]
    q, p = q0, p0
    for _ in range(int(6.0 / eps)):
        q, p = m09.leapfrog(grad, q, p, eps, 1, jnp.ones(2))
        qs.append(q)
    qs = jnp.stack(qs)
    plt.plot(qs[:, 0], qs[:, 1], ".-", ms=3, color=c, label=f"eps={eps}")
xs = jnp.linspace(-3, 3, 100); X, Y = jnp.meshgrid(xs, xs)
Z = jax.vmap(jax.vmap(lambda a, b: m09.gaussian_2d_log_prob(jnp.array([a, b]))))(X, Y)
plt.contour(X, Y, Z, levels=12, colors="gray", linewidths=0.5); plt.legend(); plt.title("Leapfrog trajectories, T = 6"); plt.show()

# %%
H = lambda q, p: -m09.gaussian_2d_log_prob(q) + 0.5 * jnp.sum(p**2)
epss = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
errs = [abs(float(H(*m09.leapfrog(grad, q0, p0, e, int(round(2.0 / e)), jnp.ones(2))) - H(q0, p0))) for e in epss]
plt.loglog(epss, errs, "o-", label="|energy error|"); plt.loglog(epss, errs[0] * (epss / epss[0]) ** 2, "--", label="eps^2")
plt.xlabel("step size"); plt.legend(); plt.title("Energy error is second order"); plt.show()

# %%
key = jax.random.key(0)
rw, acc = m09.random_walk_mh(m09.gaussian_2d_log_prob, key, jnp.zeros(2), 2000, 1.0)
hs, info = m09.hmc(m09.gaussian_2d_log_prob, key, jnp.zeros(2), 2000, 0.3, 8)
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(rw[:, 1]); ax[0].set_title(f"RWMH trace, coordinate 2 (accept {float(acc):.2f})")
ax[1].plot(hs[:, 1]); ax[1].set_title(f"HMC trace, coordinate 2 (accept {float(info.accept.mean()):.2f})")
ax[2].plot(info.energy_error); ax[2].set_title("HMC energy error per transition"); plt.tight_layout(); plt.show()
