"""Module 14 — Amortised variational inference: a VAE through the PPL.

The variational autoencoder is Module 8's ELBO with two changes: the model has a local
latent z_i per datum and a neural decoder p(x_i | z_i), and the guide's parameters are
not free per datum but *amortised*: an encoder network maps x_i to (mu_i, sigma_i).
Everything is written with the Module 12 primitives, so the ELBO is computed from two
traces exactly as a generic SVI engine would compute it.

Dataset: data.gene_programs() — 2000 cells x 64 genes, binary presence/absence, with six
latent transcriptional programs each switching on a block of genes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .m08_bbvi import adam_init, adam_update
from .m12_handlers import Bernoulli, Normal, param, replay, sample, seed, trace

Array = jax.Array
Params = dict


# ======================================================================================
# Step 1: a hand-written MLP
# ======================================================================================
def init_mlp(key: Array, sizes: list[int]) -> list[dict[str, Array]]:
    """Glorot-uniform weights and zero biases for a chain of dense layers.
    sizes = [in, h1, ..., out]. Returns a list of {"w": [in_i, out_i], "b": [out_i]}."""
    raise NotImplementedError  # Module 14, Step 1


def mlp(params: list[dict[str, Array]], x: Array, activation=jnp.tanh) -> Array:
    """Apply the layers with `activation` (tanh by default) between them and no
    nonlinearity on the output. Modules 15a and 15b reuse this network."""
    raise NotImplementedError  # Module 14, Step 1


# ======================================================================================
# Step 2: model and guide as programs
# ======================================================================================
def vae_model(params: Params, x: Array, latent_dim: int) -> None:
    """z_i ~ N(0, I_L) for each of the B rows of x; logits = decoder(z);
    x_i ~ Bernoulli(logits). The decoder weights enter through `param("dec", ...)`."""
    raise NotImplementedError  # Module 14, Step 2


def vae_guide(params: Params, x: Array, latent_dim: int) -> None:
    """q(z_i | x_i) = N(mu(x_i), diag(sigma(x_i)^2)) with (mu, raw_sigma) = encoder(x_i)
    split along the last axis and sigma = softplus(raw_sigma) + 1e-4."""
    raise NotImplementedError  # Module 14, Step 2


# ======================================================================================
# Step 3: the ELBO from two traces
# ======================================================================================
def elbo(key: Array, params: Params, x: Array, latent_dim: int) -> Array:
    """Single-sample ELBO per datum (averaged over the batch):
    run the guide under trace+seed, replay the model against that trace, then
    sum(model log_probs) - sum(guide log_probs), divided by B."""
    raise NotImplementedError  # Module 14, Step 3


def elbo_analytic_kl(key: Array, params: Params, x: Array, latent_dim: int) -> Array:
    """Same objective with the Gaussian KL in closed form:
    E_q[log p(x|z)] - sum_l KL(N(mu_l, s_l^2) || N(0, 1)), per datum."""
    raise NotImplementedError  # Module 14, Step 3


# ======================================================================================
# Step 4: training
# ======================================================================================
def init_params(key: Array, n_features: int, latent_dim: int, hidden: int = 64) -> Params:
    k_enc, k_dec = jax.random.split(key)
    return {"enc": init_mlp(k_enc, [n_features, hidden, 2 * latent_dim]), "dec": init_mlp(k_dec, [latent_dim, hidden, n_features])}


def independent_bernoulli_baseline(X: Array) -> Array:
    """Per-datum log-likelihood of the best model with independent genes:
    sum_j [p_j log p_j + (1 - p_j) log(1 - p_j)] with p_j the column means (clipped to
    [1e-6, 1 - 1e-6]). This is the number the VAE has to beat."""
    raise NotImplementedError  # Module 14, Step 4


def train_vae(key: Array, X: Array, latent_dim: int = 4, n_steps: int = 1500, batch_size: int = 128, lr: float = 1e-3, hidden: int = 64) -> tuple[Params, Array]:
    """Adam ascent on the analytic-KL ELBO with minibatches drawn by randint, in
    lax.scan. Returns (params, elbo_trace [n_steps])."""
    raise NotImplementedError  # Module 14, Step 4


# ======================================================================================
# Step 5: using the fitted model
# ======================================================================================
def latent_means(params: Params, X: Array, latent_dim: int) -> Array:
    """Encoder means mu(x) for every row of X: [N, latent_dim]."""
    raise NotImplementedError  # Module 14, Step 5


def reconstruct(params: Params, x: Array, key: Array, latent_dim: int) -> Array:
    """Bernoulli means of the decoder at one posterior sample z ~ q(z | x): [B, 64]."""
    raise NotImplementedError  # Module 14, Step 5


def program_separation(latents: Array, programs: Array) -> float:
    """Mean distance between latent means of cells with identical program indicators,
    divided by the mean distance between random pairs. Below 1 means the latent space
    groups cells by program."""
    raise NotImplementedError  # Module 14, Step 5
