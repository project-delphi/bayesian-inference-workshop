# %% [markdown]
# # Module 14 — VAE through the PPL: exploration
# Training curve against the independent-gene baseline, latent space coloured by program, and reconstructions as 8x8 gene grids.

# %%
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from notebooks._impl import impl
m14 = impl("m14_vae"); data = impl("data")
X_np, Z_np, masks = data.gene_programs(); X = jnp.asarray(X_np); L = 4

# %%
params, trace_ = m14.train_vae(jax.random.key(0), X, latent_dim=L, n_steps=3000)
plt.plot(np.convolve(np.asarray(trace_), np.ones(50) / 50, mode="valid"), label="minibatch ELBO (smoothed)")
plt.axhline(float(m14.independent_bernoulli_baseline(X)), color="k", ls="--", label="independent-gene baseline")
plt.legend(); plt.xlabel("step"); plt.ylabel("nats per cell"); plt.show()

# %%
lat = np.asarray(m14.latent_means(params, X, L))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for j in range(3):
    ax[j].scatter(lat[:, 0], lat[:, 1], c=Z_np[:, j], s=4, cmap="coolwarm"); ax[j].set_title(f"latent dims 0,1 coloured by program {j}")
plt.tight_layout(); plt.show()
print("program separation:", m14.program_separation(jnp.asarray(lat[:600]), jnp.asarray(Z_np[:600])))

# %%
xb = X[:6]; recon = m14.reconstruct(params, xb, jax.random.key(1), L)
fig, ax = plt.subplots(2, 6, figsize=(12, 4))
for i in range(6):
    ax[0, i].imshow(np.asarray(xb[i]).reshape(8, 8), cmap="gray_r"); ax[0, i].axis("off")
    ax[1, i].imshow(np.asarray(recon[i]).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1); ax[1, i].axis("off")
ax[0, 0].set_title("cell (64 genes as 8x8)", loc="left"); ax[1, 0].set_title("decoder means", loc="left"); plt.show()
