# %% [markdown]
# # Module 1 — JAX warm-up: exploration
# Newton trajectories on Rosenbrock, mixture samples, and the AR(1) path.

# %%
import jax, jax.numpy as jnp, matplotlib.pyplot as plt
from notebooks._impl import impl
m01 = impl("m01_jax")

# %%
rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
x_final, traj = m01.newton_minimize(rosen, jnp.array([-1.2, 1.0]), 12)
xs = jnp.linspace(-1.5, 1.5, 200); ys = jnp.linspace(-0.5, 1.5, 200)
X, Y = jnp.meshgrid(xs, ys)
Z = jax.vmap(jax.vmap(lambda a, b: rosen(jnp.array([a, b]))))(X, Y)
plt.contour(X, Y, jnp.log(Z + 1e-3), levels=30)
plt.plot(traj[:, 0], traj[:, 1], "o-", ms=3)
plt.title("Newton iterates on Rosenbrock"); plt.show()

# %%
x, z = m01.sample_gaussian_mixture(jax.random.key(0), jnp.array([0.2, 0.5, 0.3]), jnp.array([-3.0, 0.0, 4.0]), jnp.array([0.5, 1.0, 0.7]), 20000)
plt.hist(x, bins=120, density=True); plt.title("Mixture samples"); plt.show()

# %%
path = m01.ar1_simulate(jax.random.key(1), 0.95, 0.3, 500)
plt.plot(path); plt.title("AR(1) path, phi = 0.95"); plt.show()
