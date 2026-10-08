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
    # [m14 step 1]
    layers = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        lim = jnp.sqrt(6.0 / (n_in + n_out))
        layers.append({"w": jax.random.uniform(k, (n_in, n_out), minval=-lim, maxval=lim), "b": jnp.zeros(n_out)})
    return layers


def mlp(params: list[dict[str, Array]], x: Array) -> Array:
    """Apply the layers with tanh between them and no nonlinearity on the output."""
    # [m14 step 1]
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


# ======================================================================================
# Step 2: model and guide as programs
# ======================================================================================
def vae_model(params: Params, x: Array, latent_dim: int) -> None:
    """z_i ~ N(0, I_L) for each of the B rows of x; logits = decoder(z);
    x_i ~ Bernoulli(logits). The decoder weights enter through `param("dec", ...)`."""
    # [m14 step 2]
    dec = param("dec", params["dec"])
    B = x.shape[0]
    z = sample("z", Normal(jnp.zeros((B, latent_dim)), 1.0))
    logits = mlp(dec, z)
    sample("x", Bernoulli(logits=logits), obs=x)


def vae_guide(params: Params, x: Array, latent_dim: int) -> None:
    """q(z_i | x_i) = N(mu(x_i), diag(sigma(x_i)^2)) with (mu, raw_sigma) = encoder(x_i)
    split along the last axis and sigma = softplus(raw_sigma) + 1e-4."""
    # [m14 step 2]
    enc = param("enc", params["enc"])
    h = mlp(enc, x)
    mu, raw = h[:, :latent_dim], h[:, latent_dim:]
    sample("z", Normal(mu, jax.nn.softplus(raw) + 1e-4))


# ======================================================================================
# Step 3: the ELBO from two traces
# ======================================================================================
def elbo(key: Array, params: Params, x: Array, latent_dim: int) -> Array:
    """Single-sample ELBO per datum (averaged over the batch):
    run the guide under trace+seed, replay the model against that trace, then
    sum(model log_probs) - sum(guide log_probs), divided by B."""
    # [m14 step 3]
    guide_tr = trace(seed(vae_guide, key)).get_trace(params, x, latent_dim)
    model_tr = trace(replay(vae_model, guide_tr)).get_trace(params, x, latent_dim)
    lp_model = sum(jnp.sum(s["fn"].log_prob(s["value"])) for s in model_tr.values() if s["type"] == "sample")
    lp_guide = sum(jnp.sum(s["fn"].log_prob(s["value"])) for s in guide_tr.values() if s["type"] == "sample")
    return (lp_model - lp_guide) / x.shape[0]


def elbo_analytic_kl(key: Array, params: Params, x: Array, latent_dim: int) -> Array:
    """Same objective with the Gaussian KL in closed form:
    E_q[log p(x|z)] - sum_l KL(N(mu_l, s_l^2) || N(0, 1)), per datum."""
    # [m14 step 3]
    guide_tr = trace(seed(vae_guide, key)).get_trace(params, x, latent_dim)
    q = guide_tr["z"]["fn"]
    mu, s = q.loc, q.scale
    kl = jnp.sum(0.5 * (mu**2 + s**2 - 1.0) - jnp.log(s))
    model_tr = trace(replay(vae_model, guide_tr)).get_trace(params, x, latent_dim)
    log_lik = jnp.sum(model_tr["x"]["fn"].log_prob(model_tr["x"]["value"]))
    return (log_lik - kl) / x.shape[0]


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
    # [m14 step 4]
    p = jnp.clip(X.mean(0), 1e-6, 1 - 1e-6)
    return jnp.sum(p * jnp.log(p) + (1 - p) * jnp.log1p(-p))


def train_vae(key: Array, X: Array, latent_dim: int = 4, n_steps: int = 1500, batch_size: int = 128, lr: float = 1e-3, hidden: int = 64) -> tuple[Params, Array]:
    """Adam ascent on the analytic-KL ELBO with minibatches drawn by randint, in
    lax.scan. Returns (params, elbo_trace [n_steps])."""
    # [m14 step 4]
    k_init, k_run = jax.random.split(key)
    params = init_params(k_init, X.shape[1], latent_dim, hidden)
    state = adam_init(params)
    N = X.shape[0]

    def body(carry, k):
        params, state = carry
        k_idx, k_z = jax.random.split(k)
        xb = X[jax.random.randint(k_idx, (batch_size,), 0, N)]
        value, grads = jax.value_and_grad(elbo_analytic_kl, argnums=1)(k_z, params, xb, latent_dim)
        params, state = adam_update(grads, state, params, lr)
        return (params, state), value

    (params, _), trace_ = lax.scan(body, (params, state), jax.random.split(k_run, n_steps))
    return params, trace_


# ======================================================================================
# Step 5: using the fitted model
# ======================================================================================
def latent_means(params: Params, X: Array, latent_dim: int) -> Array:
    """Encoder means mu(x) for every row of X: [N, latent_dim]."""
    # [m14 step 5]
    return mlp(params["enc"], X)[:, :latent_dim]


def reconstruct(params: Params, x: Array, key: Array, latent_dim: int) -> Array:
    """Bernoulli means of the decoder at one posterior sample z ~ q(z | x): [B, 64]."""
    # [m14 step 5]
    guide_tr = trace(seed(vae_guide, key)).get_trace(params, x, latent_dim)
    z = guide_tr["z"]["value"]
    return jax.nn.sigmoid(mlp(params["dec"], z))


def program_separation(latents: Array, programs: Array) -> float:
    """Mean distance between latent means of cells with identical program indicators,
    divided by the mean distance between random pairs. Below 1 means the latent space
    groups cells by program."""
    # [m14 step 5]
    d = jnp.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=-1)
    same = jnp.all(programs[:, None, :] == programs[None, :, :], axis=-1)
    off = ~jnp.eye(latents.shape[0], dtype=bool)
    within = jnp.sum(d * (same & off)) / jnp.sum(same & off)
    overall = jnp.sum(d * off) / jnp.sum(off)
    return float(within / overall)
