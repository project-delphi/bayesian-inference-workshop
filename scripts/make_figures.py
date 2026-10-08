#!/usr/bin/env python
"""Render every static figure and GIF used by the site into site/figures/.

Each figure is one function registered in FIGURES. Figures are computed from the
reference implementations in solutions/ with fixed PRNG keys, so the output is
deterministic. Slow steps (training, multi-chain HMC) are cached under
site/figures/_cache/ so single figures can be re-rendered quickly.

Usage:
    python scripts/make_figures.py              # everything
    python scripts/make_figures.py m02b_grid_posterior m09_leapfrog.gif
    python scripts/make_figures.py --list
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "site" / "figures"
CACHE = OUT / "_cache"

from solutions import data  # noqa: E402

ORANGE, INK, BLUE, RED, GREY = "#c2410c", "#1f2937", "#2563eb", "#dc2626", "#9ca3af"
LIGHT_ORANGE = "#f4a582"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=[ORANGE, INK, BLUE, "#16a34a", "#7c3aed", GREY]),
    }
)
FIGURES: dict[str, callable] = {}


def figure(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn

    return deco


def key(i: int) -> jax.Array:
    return jax.random.fold_in(jax.random.key(20261008), i)


def save(fig, name: str) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))
    return path


def save_gif(anim: animation.FuncAnimation, fig, name: str, fps: int) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path.relative_to(ROOT), f"({path.stat().st_size / 1e6:.2f} MB)")
    return path


def cached(name: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = compute()
    value = jax.tree_util.tree_map(lambda a: np.asarray(a) if hasattr(a, "shape") else a, value)
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def gauss_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def contour_levels_2d(log_prob, xlim, ylim, n=200):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.asarray(jax.vmap(log_prob)(pts)).reshape(X.shape)
    return X, Y, Z


# ======================================================================================
# Module 2b
# ======================================================================================
def _m02b_setup():
    from solutions import m02b_foundations as m

    x = jnp.asarray(data.qpcr_log_expression()[0])
    mu_grid = jnp.linspace(1.2, 2.4, 241)
    ll_grid = jnp.linspace(0.0, 4.5, 301)
    log_post, log_ev = m.grid_log_posterior(m.log_joint, x, mu_grid, ll_grid)
    return m, x, mu_grid, ll_grid, np.asarray(log_post), float(log_ev)


def _m02b_marginals(log_post, mu_grid, ll_grid):
    d_mu = float(mu_grid[1] - mu_grid[0])
    d_ll = float(ll_grid[1] - ll_grid[0])
    post = np.exp(log_post)
    return post.sum(0) * d_mu, post.sum(1) * d_ll  # marginal over log lambda, over mu


@figure("m02b_grid_posterior.png")
def m02b_grid_posterior():
    m, x, mu_grid, ll_grid, log_post, log_ev = _m02b_setup()
    mean, cov = m.grid_moments(jnp.asarray(log_post), mu_grid, ll_grid)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    M, L = np.meshgrid(np.asarray(mu_grid), np.asarray(ll_grid), indexing="ij")
    cs = ax.contourf(M, L, np.exp(log_post), levels=30, cmap="Oranges")
    fig.colorbar(cs, ax=ax, label="posterior density")
    ax.plot(float(mean[0]), float(mean[1]), "x", color=INK, ms=9, mew=2, label="posterior mean")
    ax.set_xlabel("mu (log2 fold change)")
    ax.set_ylabel("log lambda (log precision)")
    ax.set_title(f"Grid posterior, qPCR replicates (log evidence {log_ev:.2f})")
    ax.legend(loc="upper left")
    return save(fig, "m02b_grid_posterior.png")


def _m02b_fits():
    m, x, mu_grid, ll_grid, log_post, log_ev = _m02b_setup()
    eps = jax.random.normal(key(0), (256, 2))
    q_mean, q_log_sd, elbo = m.fit_gaussian_vi(m.log_joint, x, eps, jnp.array([1.5, 1.0]), jnp.log(jnp.array([0.3, 0.6])))
    mode, lcov = m.laplace_approx(m.log_joint, x, jnp.array([1.5, 1.0]))
    samples, acc = m.random_walk_metropolis(lambda t: m.log_joint(t, x), key(1), jnp.array([1.5, 1.0]), 20000, 0.35)
    return dict(q_mean=np.asarray(q_mean), q_sd=np.exp(np.asarray(q_log_sd)), elbo=np.asarray(elbo), mode=np.asarray(mode), lsd=np.sqrt(np.diag(np.asarray(lcov))), samples=np.asarray(samples), acc=float(acc), log_ev=log_ev)


@figure("m02b_four_routes.png")
def m02b_four_routes():
    m, x, mu_grid, ll_grid, log_post, log_ev = _m02b_setup()
    marg_ll, marg_mu = _m02b_marginals(log_post, mu_grid, ll_grid)
    f = _m02b_fits()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, grid, marg, k, name in [(axes[0], np.asarray(ll_grid), marg_ll, 1, "log lambda"), (axes[1], np.asarray(mu_grid), marg_mu, 0, "mu")]:
        ax.hist(f["samples"][2000:, k], bins=60, density=True, color=GREY, alpha=0.6, label=f"Metropolis samples (accept {f['acc']:.2f})")
        ax.plot(grid, marg, color=INK, lw=2, label="grid (exact)")
        ax.plot(grid, gauss_pdf(grid, f["q_mean"][k], f["q_sd"][k]), color=ORANGE, lw=1.8, label="Gaussian by KL (VI)")
        ax.plot(grid, gauss_pdf(grid, f["mode"][k], f["lsd"][k]), color=BLUE, lw=1.5, ls="--", label="Laplace")
        ax.set_xlabel(name)
        ax.set_ylabel("marginal posterior density")
    axes[0].set_title("Four routes to one marginal: log precision")
    axes[1].set_title("The same four routes: mean")
    axes[0].set_xlim(0.5, 4.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    return save(fig, "m02b_four_routes.png")


@figure("m02b_elbo_bound.png")
def m02b_elbo_bound():
    f = _m02b_fits()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(f["elbo"], color=ORANGE, label="ELBO of the Gaussian fit")
    ax.axhline(f["log_ev"], color=INK, ls="--", label=f"log evidence from the grid ({f['log_ev']:.2f})")
    ax.set_ylim(f["log_ev"] - 5, f["log_ev"] + 0.5)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("nats")
    ax.set_title("The ELBO is a lower bound; the gap is KL(q || posterior)")
    ax.legend(loc="lower right")
    return save(fig, "m02b_elbo_bound.png")


@figure("m02b_jacobian.png")
def m02b_jacobian():
    from scipy.stats import gamma as sp_gamma

    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 2.0, size=20000)  # shape 2, scale 2 = rate 0.5
    dist = sp_gamma(2.0, scale=2.0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.hist(lam, bins=80, density=True, color=GREY, alpha=0.7, label="20 000 samples")
    g = np.linspace(0.01, 25, 500)
    ax.plot(g, dist.pdf(g), color=INK, lw=2, label="Gamma(2, rate 0.5) density")
    ax.set_xlim(0, 25)
    ax.set_xlabel("lambda (precision)")
    ax.set_ylabel("density")
    ax.set_title("A density over lambda")
    ax.legend()
    ax = axes[1]
    phi = np.log(lam)
    ax.hist(phi, bins=80, density=True, color=GREY, alpha=0.7, label="the same samples, as log lambda")
    p = np.linspace(-5, 4, 600)
    correct = dist.pdf(np.exp(p)) * np.exp(p)
    wrong = dist.pdf(np.exp(p))
    I_c, I_w = np.trapezoid(correct, p), np.trapezoid(wrong, p)
    ax.plot(p, correct, color=INK, lw=2, label=f"with Jacobian: integrates to {I_c:.2f}")
    ax.plot(p, wrong, color=RED, lw=1.8, ls="--", label=f"without Jacobian: integrates to {I_w:.2f}")
    ax.set_xlabel("phi = log lambda")
    ax.set_title("Mass is conserved; the Jacobian keeps it so")
    ax.legend(fontsize=8)
    return save(fig, "m02b_jacobian.png")


@figure("m02b_shrinkage.png")
def m02b_shrinkage():
    lam, n, xbar = 8.16, 12, 1.8077
    mu = np.linspace(0.5, 2.5, 800)
    lik = gauss_pdf(mu, xbar, 1 / np.sqrt(n * lam))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.fill_between(mu, lik, color=GREY, alpha=0.35, label="likelihood (as a function of mu)")
    for sd0, col, name in [(10.0, BLUE, "N(0, 10^2)"), (0.5, ORANGE, "N(0, 0.5^2)")]:
        prec = 1 / sd0**2 + n * lam
        mean = (n * lam * xbar) / prec
        ax.plot(mu, gauss_pdf(mu, 0.0, sd0), color=col, ls=":", lw=1.5, label=f"prior {name}")
        ax.plot(mu, gauss_pdf(mu, mean, prec**-0.5), color=col, lw=2, label=f"posterior, mean {mean:.4f}")
        ax.axvline(mean, color=col, lw=0.8, alpha=0.6)
    ax.set_xlabel("mu (log2 fold change)")
    ax.set_ylabel("density")
    ax.set_title("Shrinkage: a strong prior pulls the posterior toward zero")
    ax.legend(fontsize=8, loc="upper left")
    return save(fig, "m02b_shrinkage.png")


@figure("m02b_metropolis.gif")
def m02b_metropolis_gif():
    m, x, mu_grid, ll_grid, log_post, log_ev = _m02b_setup()
    marg_ll, _ = _m02b_marginals(log_post, mu_grid, ll_grid)
    lj = jax.jit(lambda t: m.log_joint(t, x))
    rng = np.random.default_rng(7)
    n_steps, step = 1800, 0.35
    theta = np.array([1.5, 1.0])
    lp = float(lj(jnp.asarray(theta)))
    states, props, accs = [], [], []
    for _ in range(n_steps):
        prop = theta + step * rng.normal(size=2)
        lp_prop = float(lj(jnp.asarray(prop)))
        acc = np.log(rng.uniform()) < lp_prop - lp
        if acc:
            theta, lp = prop, lp_prop
        states.append(theta.copy())
        props.append(prop)
        accs.append(acc)
    states, props, accs = np.array(states), np.array(props), np.array(accs)

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3, 2]})
    M, L = np.meshgrid(np.asarray(mu_grid), np.asarray(ll_grid), indexing="ij")
    ax.contour(M, L, np.exp(log_post), levels=8, colors=GREY, linewidths=0.8)
    ax.set_xlim(1.2, 2.4)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("mu")
    ax.set_ylabel("log lambda")
    trail = LineCollection([], colors=ORANGE, linewidths=1.0)
    ax.add_collection(trail)
    cur = ax.plot([], [], "o", color=INK, ms=5)[0]
    pr = ax.plot([], [], "o", mfc="none", ms=8, mew=1.5)[0]
    axh.plot(np.asarray(ll_grid), marg_ll, color=INK, lw=2, label="grid marginal")
    axh.set_xlim(0, 4.5)
    axh.set_ylim(0, 1.3)
    axh.set_xlabel("log lambda")
    axh.set_ylabel("density")
    axh.legend(loc="upper right", fontsize=8)
    per = 20

    def update(i):
        k = per * (i + 1)
        seg = states[max(0, k - 200) : k]
        segs = np.stack([seg[:-1], seg[1:]], axis=1)
        trail.set_segments(segs)
        alphas = np.linspace(0.1, 1.0, len(segs))
        trail.set_color([(0.76, 0.25, 0.05, a) for a in alphas])
        cur.set_data([states[k - 1, 0]], [states[k - 1, 1]])
        pr.set_data([props[k - 1, 0]], [props[k - 1, 1]])
        pr.set_color("#16a34a" if accs[k - 1] else RED)
        for patch in list(axh.patches):
            patch.remove()
        axh.hist(states[:k, 1], bins=np.linspace(0, 4.5, 46), density=True, color=ORANGE, alpha=0.5)
        ax.set_title(f"Metropolis step {k}, acceptance {accs[:k].mean():.2f}", fontsize=10)
        axh.set_title("log lambda so far", fontsize=10)
        return trail, cur, pr

    anim = animation.FuncAnimation(fig, update, frames=n_steps // per, blit=False)
    return save_gif(anim, fig, "m02b_metropolis.gif", fps=8)


# ======================================================================================
# Module 4
# ======================================================================================
@figure("m04_kl_asymmetry.png")
def m04_kl_asymmetry():
    x = np.linspace(-8, 8, 1200)
    p, q = gauss_pdf(x, 0, 1), gauss_pdf(x, 0, 3)
    kl_pq = 0.5 * (1 / 9 - 1 + np.log(9))
    kl_qp = 0.5 * (9 - 1 - np.log(9))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, integrand, val, title, col in [
        (axes[0], p * np.log(p / q), kl_pq, "KL(N(0,1) || N(0,9))", ORANGE),
        (axes[1], q * np.log(q / p), kl_qp, "KL(N(0,9) || N(0,1))", BLUE),
    ]:
        ax.plot(x, p, color=INK, lw=1, label="N(0, 1)")
        ax.plot(x, q, color=GREY, lw=1, label="N(0, 9)")
        ax.fill_between(x, integrand, color=col, alpha=0.5, label="integrand")
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_title(f"{title} = {val:.3f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("density / integrand")
    axes[0].legend(fontsize=8)
    return save(fig, "m04_kl_asymmetry.png")


@figure("m04_kl_bimodal.png")
def m04_kl_bimodal():
    x = np.linspace(-7, 7, 4001)
    dx = x[1] - x[0]
    p = 0.5 * gauss_pdf(x, -2, 0.6) + 0.5 * gauss_pdf(x, 2, 0.6)

    def kl_qp(mu, s):
        q = gauss_pdf(x, mu, s)
        return np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx

    best = min(((kl_qp(mu, s), mu, s) for mu in np.linspace(-3, 3, 121) for s in np.linspace(0.2, 3.0, 57)))
    _, mu_ms, s_ms = best
    mu_mm, s_mm = 0.0, np.sqrt(0.36 + 4.0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.fill_between(x, p, color=GREY, alpha=0.4, label="target p (bimodal)")
    ax.plot(x, gauss_pdf(x, mu_ms, s_ms), color=ORANGE, lw=2, label=f"min KL(q||p): mode-seeking, sd {s_ms:.2f}")
    ax.plot(x, gauss_pdf(x, mu_mm, s_mm), color=BLUE, lw=2, label=f"min KL(p||q): mass-covering, sd {s_mm:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("The direction of KL chooses the failure mode")
    ax.legend(fontsize=8)
    return save(fig, "m04_kl_bimodal.png")


@figure("m04_bregman_tangent.png")
def m04_bregman_tangent():
    A = lambda e: np.log1p(np.exp(e))
    pA, pB = 0.062, 0.079
    ep, eq = np.log(pA / (1 - pA)), np.log(pB / (1 - pB))
    mu_p = pA
    tangent = lambda e: A(ep) + (e - ep) * mu_p
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    e = np.linspace(-3.3, -1.9, 400)
    ax.plot(e, A(e), color=INK, lw=2, label="A(eta) = log(1 + e^eta)")
    ax.plot(e, tangent(e), color=GREY, lw=1.5, label="tangent at eta_p")
    ax.plot([eq, eq], [tangent(eq), A(eq)], color=ORANGE, lw=3, label=f"gap = KL(p||q) = {A(eq) - tangent(eq):.5f}")
    ax.plot([ep], [A(ep)], "o", color=INK)
    ax.plot([eq], [A(eq)], "o", color=ORANGE)
    ax.annotate("eta_p", (ep, A(ep)), xytext=(ep - 0.25, A(ep) + 0.012), fontsize=9)
    ax.annotate("eta_q", (eq, A(eq)), xytext=(eq + 0.05, A(eq) - 0.008), fontsize=9)
    ax.set_xlabel("natural parameter eta")
    ax.set_ylabel("A(eta)")
    ax.set_title("KL = height of A above its tangent (zoomed)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    e = np.linspace(-4, 1.5, 400)
    ax.plot(e, A(e), color=INK, lw=2, label="A(eta)")
    ax.plot(e, tangent(e), color=GREY, lw=1.5, label=f"tangent at eta_p, slope mu_p = {mu_p:.3f}")
    inter = tangent(0.0)
    ax.plot([0], [inter], "o", color=ORANGE)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.annotate(f"y-intercept = {inter:.4f}\n= -A*(mu_p) = entropy H(p)", (0, inter), xytext=(-3.8, 0.9), fontsize=9, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlabel("natural parameter eta")
    ax.set_title("A* is the intercept of the same tangent", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return save(fig, "m04_bregman_tangent.png")


@figure("m04_newton_damping.png")
def m04_newton_damping():
    from solutions.m03_expfam import Gamma

    fam = Gamma()
    mu = fam.mean_params(fam.to_natural(3.5, 1.7))
    eta0 = fam.to_natural(jnp.asarray(1.0), jnp.asarray(1.0))
    steps = 0.5 ** jnp.arange(11)
    obj = lambda e: fam.log_partition(e) - jnp.dot(e, mu)
    path = [np.asarray(eta0)]
    eta = eta0
    full_step = None
    for _ in range(12):
        f0 = obj(eta)
        d = -jnp.linalg.solve(fam.fisher(eta), fam.mean_params(eta) - mu)
        if full_step is None:
            full_step = np.asarray(eta + d)
        cands = eta[None, :] + steps[:, None] * d[None, :]
        fc = jax.vmap(obj)(cands)
        ok = jnp.isfinite(fc) & (fc <= f0)
        eta = jnp.where(jnp.any(ok), cands[jnp.argmax(ok)], eta)
        path.append(np.asarray(eta))
    path = np.array(path)
    target = np.asarray(fam.to_natural(3.5, 1.7))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.axhspan(0, 1.0, color=RED, alpha=0.08)
    ax.axhline(0, color=RED, lw=1.2, label="domain boundary eta_2 = 0 (rate = 0)")
    ax.annotate("", xy=full_step, xytext=path[0], arrowprops=dict(arrowstyle="->", color=RED, ls="--", lw=1.5))
    ax.plot([], [], color=RED, ls="--", label="undamped Newton step (leaves the domain, A = nan)")
    ax.plot(path[:, 0], path[:, 1], "o-", color=ORANGE, ms=4, label="damped Newton path")
    ax.plot(*target, "*", color=INK, ms=12, label="solution eta(mu) = (2.5, -1.7)")
    ax.plot(*path[0], "s", color=INK, ms=7, label="start (alpha, beta) = (1, 1)")
    ax.set_xlabel("eta_1 = alpha - 1")
    ax.set_ylabel("eta_2 = -beta")
    ax.set_title("Inverting the mean map for the Gamma: why Newton needs damping")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(min(path[:, 1].min(), -2.2) - 0.3, 0.6)
    return save(fig, "m04_newton_damping.png")


# ======================================================================================
# Module 6
# ======================================================================================
@figure("m06_meanfield_ellipse.png")
def m06_meanfield_ellipse():
    rho = 0.9
    xs = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(xs, xs)
    Z_true = np.exp(-0.5 * (X**2 - 2 * rho * X * Y + Y**2) / (1 - rho**2))
    v = 1 - rho**2
    Z_mf = np.exp(-0.5 * (X**2 + Y**2) / v)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.contour(X, Y, Z_true, levels=[np.exp(-2), np.exp(-0.5)], colors=INK, linewidths=2)
    ax.contour(X, Y, Z_mf, levels=[np.exp(-2), np.exp(-0.5)], colors=ORANGE, linewidths=2)
    ax.plot([], [], color=INK, lw=2, label="posterior, rho = 0.9, marginal sd 1")
    ax.plot([], [], color=ORANGE, lw=2, label=f"mean-field optimum, sd {np.sqrt(v):.2f}")
    ax.set_aspect("equal")
    ax.set_xlabel("theta_1")
    ax.set_ylabel("theta_2")
    ax.set_title("Mean-field under KL(q||p): means right, variances too small")
    ax.legend(fontsize=8, loc="upper left")
    return save(fig, "m06_meanfield_ellipse.png")


def _m06_setup():
    from solutions import m06_cavi as m

    x, labels, means, w = data.single_cell_embedding()
    x = jnp.asarray(x)
    prior = m.GMMPrior(1.0, 3.0, 0.35)
    return m, x, labels, means, w, prior


CLUSTER_COLORS = np.array([[0.76, 0.25, 0.05], [0.15, 0.39, 0.92], [0.09, 0.64, 0.29], [0.49, 0.23, 0.93]])


@figure("m06_cavi_fit.png")
def m06_cavi_fit():
    m, x, labels, means, w, prior = _m06_setup()
    r0 = m.init_responsibilities(key(230), x, 4)
    q, trace = m.cavi(x, r0, prior, 30)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    assign = np.asarray(jnp.argmax(q.r, axis=1))
    ax.scatter(x[:, 0], x[:, 1], c=CLUSTER_COLORS[assign], s=8, alpha=0.6)
    ax.scatter(q.m[:, 0], q.m[:, 1], c=CLUSTER_COLORS, s=160, edgecolor=INK, marker="o", label="fitted centres m_k")
    ax.scatter(means[:, 0], means[:, 1], c=INK, marker="x", s=80, label="planted centres")
    ax.set_xlabel("embedding dimension 1")
    ax.set_ylabel("embedding dimension 2")
    ax.set_title("Cells coloured by argmax responsibility")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.plot(np.arange(1, 31), np.asarray(trace), "o-", color=ORANGE, ms=3)
    ax.set_xlabel("CAVI sweep")
    ax.set_ylabel("ELBO (nats)")
    ax.set_title(f"ELBO never decreases; final {float(trace[-1]):.1f}")
    return save(fig, "m06_cavi_fit.png")


@figure("m06_elbo_by_k.png")
def m06_elbo_by_k():
    m, x, labels, means, w, prior = _m06_setup()
    ks = (2, 3, 4, 5, 6)
    vals = np.asarray(m.elbo_by_k(key(230), x, ks, prior))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ks, vals, "o-", color=ORANGE)
    ax.axvline(4, color=GREY, ls="--", lw=1)
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.0f}", (k, v), xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("number of components K")
    ax.set_ylabel("final ELBO (nats)")
    ax.set_title("The ELBO as a model-selection score: planted K = 4")
    ax.set_xticks(ks)
    return save(fig, "m06_elbo_by_k.png")


@figure("m06_cavi_sweeps.gif")
def m06_cavi_sweeps_gif():
    m, x, labels, means, w, prior = _m06_setup()
    r = jax.random.dirichlet(key(233), jnp.ones(4), (x.shape[0],))  # near-uniform: takes ~7 sweeps
    alpha = m.update_dirichlet(r, prior.alpha0)
    mm, s2 = m.update_gaussian_means(x, r, prior.sigma0, prior.sigma)
    q = m.VarParams(r, alpha, mm, s2)
    states = [(np.asarray(q.r), np.asarray(q.m), float(m.elbo(x, q, prior)))]
    step = jax.jit(lambda q: m.cavi_step(x, q, prior))
    for _ in range(15):
        q = step(q)
        states.append((np.asarray(q.r), np.asarray(q.m), float(m.elbo(x, q, prior))))
    xn = np.asarray(x)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sc = ax.scatter(xn[:, 0], xn[:, 1], s=10, alpha=0.7)
    cen = ax.scatter([], [], s=200, edgecolor=INK)
    ax.scatter(means[:, 0], means[:, 1], c=INK, marker="x", s=80)
    ax.set_xlabel("embedding dimension 1")
    ax.set_ylabel("embedding dimension 2")

    def update(i):
        r, mm, e = states[i]
        sc.set_color(np.clip(r @ CLUSTER_COLORS, 0, 1))
        cen.set_offsets(mm)
        cen.set_color(CLUSTER_COLORS)
        ax.set_title(f"CAVI sweep {i}: ELBO = {e:.1f}")
        return sc, cen

    anim = animation.FuncAnimation(fig, update, frames=len(states), blit=False)
    return save_gif(anim, fig, "m06_cavi_sweeps.gif", fps=3)


# ======================================================================================
# Module 7
# ======================================================================================
@figure("m07_estimator_hist.png")
def m07_estimator_hist():
    rng = np.random.default_rng(1)
    n = 200000
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharey="row", constrained_layout=True)
    for row, mu in enumerate([0.0, 2.0]):
        eps = rng.normal(size=n)
        z = mu + eps
        score = z**2 * (z - mu)
        a = 3.0 + mu**2  # optimal constant baseline: E[f]
        cv = (z**2 - a) * (z - mu)
        path = 2 * z
        for col, (vals, name, colr) in enumerate([(score, "score-function", ORANGE), (cv, "score-function + control variate", BLUE), (path, "pathwise", INK)]):
            ax = axes[row, col]
            ax.hist(np.clip(vals, -30, 30), bins=120, density=True, color=colr, alpha=0.7)
            ax.axvline(2 * mu, color=RED, lw=1, ls="--")
            ax.set_title(f"{name}\nmu = {mu:.0f}: variance {vals.var():.1f}", fontsize=9)
            ax.set_xlim(-30, 30)
            if row == 1 and col == 1:
                ax.set_xlabel("per-sample estimate of d E[z^2] / d mu (clipped at +-30)")
        axes[row, 0].set_ylabel("density")
    fig.suptitle("Three unbiased estimators of the same gradient (true value dashed red)")
    return save(fig, "m07_estimator_hist.png")


@figure("m07_variance_scaling.png")
def m07_variance_scaling():
    from solutions import m07_gradients as m

    X, y, _ = data.saas_churn()
    X, y = jnp.asarray(X), jnp.asarray(y, float)
    lj = lambda b: m.churn_log_joint(b, X, y)
    mean, cov = m.laplace(lj, jnp.zeros(5))
    params = {"loc": mean, "log_scale": 0.5 * jnp.log(jnp.diag(cov))}
    ns = (4, 16, 64, 256)
    study = cached("m07_variance_study", lambda: m.gradient_variance_study(key(330), params, lj, ns, n_repeats=120))
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for name, label, col in [("score", "score-function", ORANGE), ("score_cv", "score-function + control variate", BLUE), ("reparam", "pathwise", INK)]:
        ax.loglog(ns, study[name], "o-", color=col, label=label)
    ref = study["score"][0] * np.array(ns[0]) / np.array(ns)
    ax.loglog(ns, ref, ls=":", color=GREY, label="slope -1 (1/n)")
    ax.set_xlabel("samples per gradient estimate n")
    ax.set_ylabel("total gradient variance")
    ax.set_title("Churn posterior: variance falls as 1/n, from very different starts")
    ax.legend(fontsize=8)
    return save(fig, "m07_variance_scaling.png")


# ======================================================================================
# Module 8
# ======================================================================================
@figure("m08_bijectors.png")
def m08_bijectors():
    from solutions.m08_bbvi import Exp, Sigmoid, Softplus

    x = jnp.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for b, name, col in [(Exp(), "exp", ORANGE), (Softplus(), "softplus", BLUE), (Sigmoid(), "sigmoid", INK)]:
        axes[0].plot(x, jax.vmap(b.forward)(x), color=col, lw=2, label=name)
        axes[1].plot(x, jax.vmap(lambda v: b.log_det_jacobian(jnp.reshape(v, (1,))))(x), color=col, lw=2, label=name)
    axes[0].set_ylim(-0.2, 6)
    axes[0].set_title("forward maps from the real line")
    axes[0].set_xlabel("unconstrained x")
    axes[0].set_ylabel("constrained value")
    axes[1].set_title("log |d forward / dx|: the Jacobian term")
    axes[1].set_xlabel("unconstrained x")
    axes[1].set_ylabel("log-det Jacobian")
    axes[0].legend()
    return save(fig, "m08_bijectors.png")


# ======================================================================================
# Module 9
# ======================================================================================
def _leapfrog_path(grad_lp, q, p, eps, n, inv_mass):
    from solutions.m09_hmc import leapfrog

    qs, ps = [np.asarray(q)], [np.asarray(p)]
    for _ in range(n):
        q, p = leapfrog(grad_lp, q, p, eps, 1, inv_mass)
        qs.append(np.asarray(q))
        ps.append(np.asarray(p))
    return np.array(qs), np.array(ps)


@figure("m09_leapfrog_phase.png")
def m09_leapfrog_phase():
    grad = lambda q: -q
    inv = jnp.ones(1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=2, label="exact flow (circle, H = 1/2)")
    for eps, col in [(0.1, GREY), (0.5, BLUE), (1.0, ORANGE)]:
        qs, ps = _leapfrog_path(grad, jnp.array([1.0]), jnp.array([0.0]), eps, 100, inv)
        ax.plot(qs[:, 0], ps[:, 0], "o-", color=col, ms=2.5, lw=0.8, label=f"leapfrog eps = {eps}")
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("position q")
    ax.set_ylabel("momentum p")
    ax.set_title("Stable: a nearby shadow energy is conserved", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42))
    ax = axes[1]
    qs, ps = _leapfrog_path(grad, jnp.array([1.0]), jnp.array([0.0]), 2.1, 12, inv)
    ax.plot(np.arange(13), np.abs(qs[:, 0]), "o-", color=RED, ms=4, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("leapfrog step")
    ax.set_ylabel("|q|")
    ax.set_title("eps = 2.1 > 2: unstable, |q| grows every step", fontsize=10)
    return save(fig, "m09_leapfrog_phase.png")


def _gauss2d_contours(ax):
    from solutions.m09_hmc import gaussian_2d_log_prob

    X, Y, Z = contour_levels_2d(gaussian_2d_log_prob, (-4, 4), (-5, 5))
    ax.contour(X, Y, np.exp(Z), levels=6, colors=GREY, linewidths=0.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")


@figure("m09_rwm_vs_hmc.png")
def m09_rwm_vs_hmc():
    from solutions.m09_hmc import gaussian_2d_log_prob, hmc, random_walk_mh

    grad = jax.grad(gaussian_2d_log_prob)
    rw, acc_rw = random_walk_mh(gaussian_2d_log_prob, key(900), jnp.zeros(2), 2000, 0.5)
    _, info = hmc(gaussian_2d_log_prob, key(901), jnp.zeros(2), 2000, 0.25, 12)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    _gauss2d_contours(axes[0])
    rwn = np.asarray(rw)[:60]
    axes[0].plot(rwn[:, 0], rwn[:, 1], "o-", color=ORANGE, ms=3, lw=0.8)
    axes[0].set_title(f"Random walk: 60 steps of scale 0.5 (acceptance {float(acc_rw):.2f})")
    _gauss2d_contours(axes[1])
    q = jnp.zeros(2)
    rng = np.random.default_rng(3)
    for i in range(8):
        p = jnp.asarray(rng.normal(size=2))
        qs, ps = _leapfrog_path(grad, q, p, 0.25, 12, jnp.ones(2))
        axes[1].plot(qs[:, 0], qs[:, 1], "-", color=BLUE, lw=0.9, alpha=0.8)
        axes[1].plot(qs[-1, 0], qs[-1, 1], "o", color=ORANGE, ms=5)
        q = jnp.asarray(qs[-1])
    axes[1].plot([], [], "-", color=BLUE, label="leapfrog trajectory (12 steps)")
    axes[1].plot([], [], "o", color=ORANGE, label="accepted state")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_title(f"HMC: 8 trajectories (acceptance {float(info.accept.mean()):.2f})")
    return save(fig, "m09_rwm_vs_hmc.png")


@figure("m09_funnel_prior.png")
def m09_funnel_prior():
    rng = np.random.default_rng(5)
    n = 20000
    mu = rng.normal(0, 5, n)
    tau = np.abs(rng.normal(0, 5, n))
    eta = rng.normal(size=n)
    theta = mu + tau * eta
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].scatter(theta - mu, np.log(tau), s=2, alpha=0.3, color=ORANGE)
    axes[0].set_xlabel("theta_1 - mu  (centred coordinate)")
    axes[0].set_ylabel("log tau")
    axes[0].set_title("Centred: theta's width is tau itself")
    axes[1].scatter(eta, np.log(tau), s=2, alpha=0.3, color=BLUE)
    axes[1].set_xlabel("eta_1  (non-centred coordinate)")
    axes[1].set_title("Non-centred: unit width whatever tau is")
    for ax in axes:
        ax.set_xlim(-15, 15)
        ax.set_ylim(-6, 3.5)
    fig.suptitle("The funnel exists in the prior: 20 000 prior draws, no data, no sampler", y=1.02)
    return save(fig, "m09_funnel_prior.png")


@figure("m09_acceptance_vs_step.png")
def m09_acceptance_vs_step():
    from solutions.m09_hmc import hmc, random_walk_mh, standard_normal_log_prob

    def compute():
        steps = np.linspace(0.2, 1.8, 9)
        acc_h = [float(hmc(standard_normal_log_prob, key(910 + i), jnp.zeros(5), 2000, float(e), 10)[1].accept.mean()) for i, e in enumerate(steps)]
        acc_r = [float(random_walk_mh(standard_normal_log_prob, key(930 + i), jnp.zeros(5), 4000, float(e))[1]) for i, e in enumerate(steps)]
        return dict(steps=steps, acc_h=np.array(acc_h), acc_r=np.array(acc_r))

    d = cached("m09_acceptance", compute)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(d["steps"], d["acc_h"], "o-", color=ORANGE, label="HMC, L = 10 leapfrog steps")
    ax.plot(d["steps"], d["acc_r"], "s-", color=INK, label="random-walk Metropolis")
    ax.axvline(2.4 / np.sqrt(5), color=GREY, ls=":", label="2.4 / sqrt(d): optimal random-walk scale")
    ax.axhline(0.8, color=GREY, lw=0.8)
    ax.set_xlabel("step size (HMC eps, or random-walk proposal scale)")
    ax.set_ylabel("mean acceptance")
    ax.set_title("5-D standard normal: acceptance against step size")
    ax.legend(fontsize=8)
    return save(fig, "m09_acceptance_vs_step.png")


@figure("m09_leapfrog.gif")
def m09_leapfrog_gif():
    grad = lambda q: -q
    inv = jnp.ones(1)
    paths = {eps: _leapfrog_path(grad, jnp.array([1.0]), jnp.array([0.0]), eps, 60, inv) for eps in (0.5, 2.1)}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    th = np.linspace(0, 2 * np.pi, 200)
    lines = {}
    dots = {}
    for ax, eps in zip(axes, (0.5, 2.1)):
        ax.plot(np.cos(th), np.sin(th), color=INK, lw=1)
        lines[eps] = ax.plot([], [], "o-", color=BLUE if eps < 2 else RED, ms=3, lw=0.8)[0]
        dots[eps] = ax.plot([], [], "o", color=ORANGE, ms=8)[0]
        lim = 1.6 if eps < 2 else 6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("position q")
    axes[0].set_ylabel("momentum p")

    def update(i):
        for eps in (0.5, 2.1):
            qs, ps = paths[eps]
            k = min(i + 1, len(qs))
            lines[eps].set_data(qs[:k, 0], ps[:k, 0])
            dots[eps].set_data([qs[k - 1, 0]], [ps[k - 1, 0]])
            dH = 0.5 * (qs[k - 1, 0] ** 2 + ps[k - 1, 0] ** 2) - 0.5
            axes[0 if eps < 2 else 1].set_title(f"eps = {eps}, step {k - 1}: energy error {dH:+.2e}" if abs(dH) < 1e3 else f"eps = {eps}, step {k - 1}: energy error {dH:+.1e}", fontsize=9)
        return list(lines.values()) + list(dots.values())

    anim = animation.FuncAnimation(fig, update, frames=60, blit=False)
    return save_gif(anim, fig, "m09_leapfrog.gif", fps=6)


@figure("m09_hmc_trajectories.gif")
def m09_hmc_trajectories_gif():
    from solutions.m09_hmc import gaussian_2d_log_prob, random_walk_mh

    grad = jax.grad(gaussian_2d_log_prob)
    n_frames = 80
    rw, _ = random_walk_mh(gaussian_2d_log_prob, key(905), jnp.zeros(2), n_frames, 0.5)
    rw = np.asarray(rw)
    rng = np.random.default_rng(11)
    q = jnp.zeros(2)
    trajs, ends, accs = [], [], []
    for _ in range(n_frames):
        p = jnp.asarray(rng.normal(size=2))
        qs, ps = _leapfrog_path(grad, q, p, 0.25, 12, jnp.ones(2))
        h0 = -float(gaussian_2d_log_prob(q)) + 0.5 * float(p @ p)
        h1 = -float(gaussian_2d_log_prob(jnp.asarray(qs[-1]))) + 0.5 * float(ps[-1] @ ps[-1])
        acc = np.log(rng.uniform()) < -(h1 - h0)
        if acc:
            q = jnp.asarray(qs[-1])
        trajs.append(qs)
        ends.append(np.asarray(q))
        accs.append(acc)
    ends = np.array(ends)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax in axes:
        _gauss2d_contours(ax)
    rw_line = axes[0].plot([], [], "o-", color=ORANGE, ms=3, lw=0.8)[0]
    traj_lines = []
    end_sc = axes[1].scatter([], [], s=18, color=ORANGE)

    def update(i):
        rw_line.set_data(rw[: i + 1, 0], rw[: i + 1, 1])
        rw_acc = np.mean(np.any(np.diff(rw[: i + 1], axis=0) != 0, axis=1)) if i > 0 else 0.0
        axes[0].set_title(f"Random walk: step {i + 1}, acceptance {rw_acc:.2f}", fontsize=9)
        for ln in traj_lines:
            ln.set_alpha(0.15)
            ln.set_color(GREY)
        qs = trajs[i]
        traj_lines.append(axes[1].plot(qs[:, 0], qs[:, 1], "-", color=BLUE, lw=1.2)[0])
        end_sc.set_offsets(ends[: i + 1])
        axes[1].set_title(f"HMC: trajectory {i + 1}, acceptance {np.mean(accs[: i + 1]):.2f}", fontsize=9)
        return [rw_line, end_sc] + traj_lines

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    return save_gif(anim, fig, "m09_hmc_trajectories.gif", fps=5)


# ======================================================================================
# Module 10
# ======================================================================================
@figure("m10_dual_averaging.png")
def m10_dual_averaging():
    from solutions.m09_hmc import hmc_step, standard_normal_log_prob
    from solutions.m10_adaptation import dual_averaging_init, dual_averaging_update

    def compute():
        inv = jnp.ones(5)
        step = jax.jit(lambda k, q, e: hmc_step(standard_normal_log_prob, k, q, e, 10, inv))
        da = dual_averaging_init(0.1)
        q = jnp.zeros(5)
        raw, avg, acc = [], [], []
        k = key(1000)
        for _ in range(1000):
            k, sub = jax.random.split(k)
            q, info = step(sub, q, jnp.exp(da.log_step))
            da = dual_averaging_update(da, info.accept_prob)
            raw.append(float(jnp.exp(da.log_step)))
            avg.append(float(jnp.exp(da.log_step_avg)))
            acc.append(float(info.accept_prob))
        return dict(raw=np.array(raw), avg=np.array(avg), acc=np.array(acc))

    d = cached("m10_dual_averaging", compute)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.semilogy(d["raw"], color=LIGHT_ORANGE, lw=0.8, label="step size tried next, exp(x_t)")
    ax.semilogy(d["avg"], color=INK, lw=2, label="averaged iterate, exp(xbar_t)")
    ax.annotate(f"final {d['avg'][-1]:.2f}", (999, d["avg"][-1]), xytext=(-70, 12), textcoords="offset points", fontsize=9)
    ax.set_xlabel("warm-up iteration")
    ax.set_ylabel("step size eps")
    ax.set_title("Dual averaging on the 5-D normal")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    run = np.cumsum(d["acc"]) / np.arange(1, 1001)
    ax.plot(run, color=ORANGE, label="running mean acceptance probability")
    ax.axhline(0.8, color=INK, ls="--", label="target 0.8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("warm-up iteration")
    ax.set_ylabel("acceptance probability")
    ax.set_title("The controller drives acceptance to the target")
    ax.legend(fontsize=8, loc="lower right")
    return save(fig, "m10_dual_averaging.png")


@figure("m10_mass_matrix.png")
def m10_mass_matrix():
    lp = lambda z: -0.5 * (z[0] ** 2 + (z[1] / 10.0) ** 2)
    grad = jax.grad(lp)
    X, Y, Z = contour_levels_2d(lp, (-4, 4), (-30, 30))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    rng = np.random.default_rng(2)
    for ax, inv, title in [(axes[0], jnp.array([1.0, 1.0]), "identity mass: z_1 zig-zags at its stability limit,\nz_2 drifts about one sd per trajectory"), (axes[1], jnp.array([1.0, 100.0]), "adapted mass M^-1 = (1, 100): both coordinates\ncomplete a smooth orbit in the same 10 steps")]:
        ax.contour(X, Y, np.exp(Z), levels=6, colors=GREY, linewidths=0.8)
        q = jnp.array([0.5, 5.0])
        for i in range(3):
            p = jnp.asarray(rng.normal(size=2)) / jnp.sqrt(inv)
            qs, _ = _leapfrog_path(grad, q, p, 1.0, 10, inv)
            ax.plot(qs[:, 0], qs[:, 1], "o-", color=[ORANGE, BLUE, "#16a34a"][i], ms=3, lw=1)
            q = jnp.asarray(qs[-1])
        ax.set_xlim(-4, 4)
        ax.set_ylim(-30, 30)
        ax.set_xlabel("z_1 (sd 1)")
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("z_2 (sd 10)")
    fig.suptitle("Three HMC trajectories, eps = 1, 10 leapfrog steps each", y=1.0)
    return save(fig, "m10_mass_matrix.png")


# ======================================================================================
# Module 11
# ======================================================================================
def _m11_funnel_runs():
    from solutions import m09_hmc as m09
    from solutions import m11_diagnostics as m11

    from tests._util import key as test_key

    def compute():
        # Exactly the setup of tests/m11: starts from key(290), chains from key(291).
        q0s = 0.1 * jax.random.normal(test_key(290), (4, m09.UPLIFT_DIM))
        out = {}
        for name, lp, to_c in [("centred", m09.uplift_centered_log_prob, lambda z: z), ("non-centred", m09.uplift_noncentered_log_prob, m09.uplift_noncentered_to_centered)]:
            ch, info, adapted = m11.run_chains(lp, test_key(291), q0s, 1000, 1000)
            div = np.asarray(~jnp.isfinite(info.energy_error) | (info.energy_error > 1000))
            out[name] = dict(chains=np.asarray(to_c(ch)), raw=np.asarray(ch), div=div, ess=float(m11.ess(ch)[1]), rhat=float(m11.split_rhat(ch)[1]), step_size=np.asarray(adapted.step_size))
        return out

    runs = cached("m11_funnel_runs", compute)
    for name, r in runs.items():
        print(f"   m11 {name}: divergences {int(r['div'].sum())}, ESS(log tau) {r['ess']:.1f}, split R-hat(log tau) {r['rhat']:.3f}, adapted eps {np.round(r['step_size'], 3)}")
    return runs


@figure("m11_funnel_divergences.png")
def m11_funnel_divergences():
    runs = _m11_funnel_runs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (name, r) in zip(axes, runs.items()):
        flat = r["chains"].reshape(-1, r["chains"].shape[-1])
        div = r["div"].reshape(-1)
        ax.scatter(flat[~div, 2], flat[~div, 1], s=3, alpha=0.3, color=ORANGE if name == "centred" else BLUE)
        ax.scatter(flat[div, 2], flat[div, 1], s=12, color=RED, label=f"divergent ({div.sum()})")
        ax.set_xlabel("theta_1 (US-East uplift, pp)")
        ax.set_title(f"{name}: ESS(log tau) = {r['ess']:.0f}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-6, 6)
    axes[0].set_ylabel("log tau")
    return save(fig, "m11_funnel_divergences.png")


@figure("m11_traces.png")
def m11_traces():
    runs = _m11_funnel_runs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (name, r) in zip(axes, runs.items()):
        for j in range(4):
            ax.plot(r["raw"][j, :, 1], lw=0.6, alpha=0.8)
        ax.set_title(f"{name}: split R-hat of log tau = {r['rhat']:.2f}")
        ax.set_xlabel("draw")
    axes[0].set_ylabel("log tau")
    return save(fig, "m11_traces.png")


@figure("m11_funnel_chains.gif")
def m11_funnel_chains_gif():
    """Four chains per parameterisation walking in (theta_1, log tau); divergent draws
    appear in red as the chains reach them. Same cached runs as the static figures."""
    runs = _m11_funnel_runs()
    per_frame, trail = 20, 160
    n_draws = runs["centred"]["chains"].shape[1]
    frames = n_draws // per_frame
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    artists = {}
    for ax, (name, r) in zip(axes, runs.items()):
        col = ORANGE if name == "centred" else BLUE
        ax.set_xlim(-6, 6)
        ax.set_ylim(-9, 2.5)
        ax.set_xlabel("theta_1 (US-East uplift, pp)")
        lines = [ax.plot([], [], lw=0.7, color=col, alpha=0.8)[0] for _ in range(4)]
        heads = ax.scatter([], [], s=30, color=INK, zorder=3)
        divs = ax.scatter([], [], s=14, color=RED, zorder=4, label="divergent")
        artists[name] = (ax, lines, heads, divs)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("log tau")

    def update(i):
        t = (i + 1) * per_frame
        for name, r in runs.items():
            ax, lines, heads, divs = artists[name]
            ch, dv = r["chains"], r["div"]
            lo = max(0, t - trail)
            for j, ln in enumerate(lines):
                ln.set_data(ch[j, lo:t, 2], ch[j, lo:t, 1])
            heads.set_offsets(np.stack([ch[:, t - 1, 2], ch[:, t - 1, 1]], 1))
            mask = dv[:, :t]
            pts = ch[:, :t][mask]
            divs.set_offsets(pts[:, [2, 1]] if len(pts) else np.zeros((0, 2)))
            ax.set_title(f"{name}\ndraw {t} of {n_draws}, {int(mask.sum())} divergent so far", fontsize=10)
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    return save_gif(anim, fig, "m11_funnel_chains.gif", fps=12)


@figure("m11_autocorrelation.png")
def m11_autocorrelation():
    from solutions.m11_diagnostics import autocorrelation

    rng = np.random.default_rng(4)
    phi, n = 0.9, 10000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    rho = np.asarray(autocorrelation(jnp.asarray(x)))
    pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(1)
    k_stop = int(np.argmax(pairs <= 0))
    tau_hat = -1 + 2 * np.minimum.accumulate(pairs[:k_stop]).sum()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    lags = np.arange(120)
    ax.bar(lags, rho[:120], color=GREY, width=1.0, label="sample autocorrelation")
    ax.plot(lags, phi**lags, color=INK, lw=1.5, label="true 0.9^t")
    ax.axvline(2 * k_stop, color=ORANGE, lw=2, label=f"Geyer truncation at lag {2 * k_stop}")
    ax.set_xlabel("lag t")
    ax.set_ylabel("rho_t")
    ax.set_title(f"AR(1), phi = 0.9: tau = 19 exactly, estimate {tau_hat:.1f}")
    ax.legend(fontsize=8)
    return save(fig, "m11_autocorrelation.png")


@figure("m11_hmc_vs_vi.png")
def m11_hmc_vs_vi():
    from solutions import m11_diagnostics as m11
    from solutions.m08_bbvi import fit

    X, y, _ = data.saas_churn()
    X, y = jnp.asarray(X), jnp.asarray(y, float)
    lj = lambda b: m11.churn_log_joint(b, X, y)

    def compute():
        chains, info, _ = m11.run_chains(lj, key(1100), jnp.zeros((4, 5)), 500, 1000)
        params, _ = fit(lj, 5, key(1101), 3000)
        return dict(chains=np.asarray(chains), loc=np.asarray(params["loc"]), scale=np.exp(np.asarray(params["log_scale"])))

    d = cached("m11_churn", compute)
    flat = d["chains"].reshape(-1, 5)
    names = ["intercept", "tenure", "active days", "tickets", "enterprise"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    idx = np.arange(5)
    ax.bar(idx - 0.2, flat.std(0, ddof=1), 0.4, color=INK, label="HMC posterior sd")
    ax.bar(idx + 0.2, d["scale"], 0.4, color=ORANGE, label="mean-field VI sd")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("posterior standard deviation")
    ax.set_title("Posterior sd: VI narrower where coordinates correlate", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(flat[::4, 0], flat[::4, 4], s=4, alpha=0.3, color=INK, label="HMC draws")
    ell = patches.Ellipse((d["loc"][0], d["loc"][4]), 4 * d["scale"][0], 4 * d["scale"][4], fill=False, color=ORANGE, lw=2, label="mean-field VI, 2 sd")
    ax.add_patch(ell)
    corr = np.corrcoef(flat[:, 0], flat[:, 4])[0, 1]
    ax.set_xlabel("intercept")
    ax.set_ylabel("enterprise coefficient")
    ax.set_title(f"HMC correlation {corr:.2f}; the product family has none", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return save(fig, "m11_hmc_vs_vi.png")


# ======================================================================================
# Module 12
# ======================================================================================
@figure("m12_message_flow.png")
def m12_message_flow():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#fff7ed", ec=INK, fs=9.5, bold=False):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=INK, text=None, side="left"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
        if text:
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(xm + (-0.15 if side == "left" else 0.15), ym, text, ha="right" if side == "left" else "left", va="center", fontsize=8.5, color=color)

    box(3.4, 5.1, 3.2, 0.7, 'sample("tau", HalfNormal(5))\nbuilds a message dict, value = None', fc="#f3f4f6", fs=8.5)
    box(3.4, 3.6, 3.2, 0.7, "handler: seed(key)  [top of stack]", bold=True)
    box(3.4, 2.1, 3.2, 0.7, "handler: trace  [bottom of stack]", bold=True)
    box(3.4, 0.5, 3.2, 0.7, "default: value = fn.sample(key)", fc="#eff6ff", fs=9)
    # down arrows (left side)
    arrow(4.0, 5.1, 4.0, 4.3, ORANGE, "process_message:\nkey = fold_in(key, 1)", "left")
    arrow(4.0, 3.6, 4.0, 2.8, ORANGE, "process_message:\n(does nothing)", "left")
    arrow(4.0, 2.1, 4.0, 1.2, ORANGE, "no value set yet,\nso draw one", "left")
    # up arrows (right side)
    arrow(6.0, 1.2, 6.0, 2.1, BLUE, "value is now a number", "right")
    arrow(6.0, 2.8, 6.0, 3.6, BLUE, 'postprocess_message:\nrecord copy under "tau"', "right")
    arrow(6.0, 4.3, 6.0, 5.1, BLUE, "postprocess_message:\n(does nothing)", "right")
    ax.text(5.0, 5.95, 'returns msg["value"] to the model', ha="center", fontsize=9, color=BLUE)
    ax.text(0.3, 4.6, "down the stack,\ninnermost first", color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.text(8.2, 1.4, "back up,\nin reverse", color=BLUE, fontsize=9.5, fontweight="bold")
    ax.set_title("The life of one sample call under trace(seed(model, key))", fontsize=11)
    return save(fig, "m12_message_flow.png")


# ======================================================================================
# Module 13
# ======================================================================================
@figure("m13_unconstrained_density.png")
def m13_unconstrained_density():
    from scipy.stats import halfnorm

    yv = np.linspace(-8, 4, 1200)
    pdf = halfnorm(scale=5).pdf
    correct = pdf(np.exp(yv)) * np.exp(yv)
    wrong = pdf(np.exp(yv))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(yv, correct, color=INK, lw=2, label=f"with log-Jacobian +y: integrates to {np.trapezoid(correct, yv):.2f}")
    ax.plot(yv, wrong, color=RED, lw=1.8, ls="--", label=f"without: integrates to {np.trapezoid(wrong, yv):.2f}")
    ax.set_xlabel("y = log tau, tau ~ HalfNormal(5)")
    ax.set_ylabel("density")
    ax.set_title("Unconstraining a HalfNormal scale: the Jacobian is not optional")
    ax.legend(fontsize=8)
    return save(fig, "m13_unconstrained_density.png")


# ======================================================================================
# Module 14
# ======================================================================================
def _m14_fit():
    from solutions import m14_vae as m

    X, Z, masks = data.gene_programs()
    X = jnp.asarray(X)
    d = cached("m14_vae", lambda: dict(zip(("params", "trace"), m.train_vae(key(110), X, latent_dim=4, n_steps=1500, batch_size=128, lr=1e-3))))
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    return m, X, Z, params, np.asarray(d["trace"])


@figure("m14_latent_space.png")
def m14_latent_space():
    m, X, Z, params, trace_ = _m14_fit()
    lat = np.asarray(m.latent_means(params, X, 4))
    sep = m.program_separation(jnp.asarray(lat[:600]), jnp.asarray(Z[:600]))
    codes = (Z[:, :3] @ np.array([4, 2, 1])).astype(int)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sc = ax.scatter(lat[:, 0], lat[:, 1], c=codes, cmap="tab10", s=6, alpha=0.7)
    ax.set_xlabel("latent mean, dimension 1")
    ax.set_ylabel("latent mean, dimension 2")
    ax.set_title(f"Encoder means coloured by programs 1-3 on/off; separation {sep:.2f}")
    fig.colorbar(sc, ax=ax, label="program indicator code (binary 1-3)")
    return save(fig, "m14_latent_space.png")


@figure("m14_reconstructions.png")
def m14_reconstructions():
    m, X, Z, params, trace_ = _m14_fit()
    xb = X[:6]
    rec = np.asarray(m.reconstruct(params, xb, key(123), 4))
    z_prior = jax.random.normal(key(124), (6, 4))
    prior_rec = np.asarray(jax.nn.sigmoid(m.mlp(params["dec"], z_prior)))
    fig, axes = plt.subplots(3, 6, figsize=(8, 4.2))
    for j in range(6):
        for i, (arr, name) in enumerate([(np.asarray(xb[j]), "data"), (rec[j], "reconstruction"), (prior_rec[j], "prior draw")]):
            ax = axes[i, j]
            ax.imshow(arr.reshape(8, 8), cmap="Oranges", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Six cells as 8x8 gene grids: data, decoder means at q(z|x), decoder at a prior z", y=0.98, fontsize=10)
    return save(fig, "m14_reconstructions.png")


@figure("m14_elbo_training.png")
def m14_elbo_training():
    m, X, Z, params, trace_ = _m14_fit()
    base = float(m.independent_bernoulli_baseline(X))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    sm = np.convolve(-trace_, np.ones(20) / 20, mode="valid")
    ax.plot(sm, color=ORANGE, label="negative ELBO per cell (20-step average)")
    ax.axhline(-base, color=INK, ls="--", label=f"independent-Bernoulli baseline {-base:.1f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("nats per cell")
    ax.set_title("The VAE must beat the model with no latent structure")
    ax.legend(fontsize=8)
    return save(fig, "m14_elbo_training.png")


@figure("m14_training.gif")
def m14_training_gif():
    """VAE training in snapshots: six cells' decoder means at the encoder mean after
    every 50 Adam steps, with the ELBO curve filling in. Re-runs train_vae's loop in
    chunks with the same key so the snapshots are the actual optimisation path."""
    from jax import lax

    from solutions import m14_vae as m
    from solutions.m08_bbvi import adam_init, adam_update

    X, Z, masks = data.gene_programs()
    X = jnp.asarray(X)
    latent_dim, batch_size, lr, chunk, n_chunks = 4, 128, 1e-3, 50, 30
    k_init, k_run = jax.random.split(key(110))
    params = m.init_params(k_init, X.shape[1], latent_dim, 64)
    state = adam_init(params)
    N = X.shape[0]

    def body(carry, k):
        params, state = carry
        k_idx, k_z = jax.random.split(k)
        xb = X[jax.random.randint(k_idx, (batch_size,), 0, N)]
        value, grads = jax.value_and_grad(m.elbo_analytic_kl, argnums=1)(k_z, params, xb, latent_dim)
        params, state = adam_update(grads, state, params, lr)
        return (params, state), value

    run_chunk = jax.jit(lambda carry, ks: lax.scan(body, carry, ks))
    keys = jax.random.split(k_run, chunk * n_chunks).reshape(n_chunks, chunk)  # same keys as train_vae's 1500 steps
    xb = X[:6]
    snaps, trace_ = [], []
    recon = lambda p: np.asarray(jax.nn.sigmoid(m.mlp(p["dec"], m.latent_means(p, xb, latent_dim))))
    snaps.append(recon(params))
    carry = (params, state)
    for c in range(n_chunks):
        carry, vals = run_chunk(carry, keys[c])
        trace_.extend(np.asarray(vals).tolist())
        snaps.append(recon(carry[0]))
    base = float(m.independent_bernoulli_baseline(X))
    trace_ = np.asarray(trace_)

    fig = plt.figure(figsize=(8.4, 3.0))
    gs = fig.add_gridspec(2, 7, width_ratios=[1] * 6 + [3.2], wspace=0.08, hspace=0.12, bottom=0.17, top=0.86, left=0.06, right=0.94)
    img_axes = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(2)]
    ax_curve = fig.add_subplot(gs[:, 6])
    ims = []
    for j in range(6):
        img_axes[0][j].imshow(np.asarray(xb[j]).reshape(8, 8), cmap="Oranges", vmin=0, vmax=1)
        ims.append(img_axes[1][j].imshow(snaps[0][j].reshape(8, 8), cmap="Oranges", vmin=0, vmax=1))
        for i in range(2):
            img_axes[i][j].set_xticks([])
            img_axes[i][j].set_yticks([])
    img_axes[0][0].set_ylabel("data", fontsize=9)
    img_axes[1][0].set_ylabel("decoder", fontsize=9)
    ax_curve.axhline(-base, color=INK, ls="--", lw=1, label=f"baseline {-base:.1f}")
    (curve,) = ax_curve.plot([], [], color=ORANGE, lw=1.2, label="negative ELBO per cell")
    ax_curve.set_xlim(0, chunk * n_chunks)
    ax_curve.set_ylim(20, 48)
    ax_curve.set_xlabel("Adam step")
    ax_curve.legend(fontsize=7, loc="upper right")
    ax_curve.yaxis.tick_right()

    def update(i):
        for j in range(6):
            ims[j].set_data(snaps[i][j].reshape(8, 8))
        t = i * chunk
        if t > 0:
            sm = np.convolve(-trace_[:t], np.ones(10) / 10, mode="valid")
            curve.set_data(np.arange(len(sm)) + 9, sm)
        fig.suptitle(f"VAE after {t} steps: six cells, their reconstructions, and the bound", fontsize=10)
        return ims + [curve]

    anim = animation.FuncAnimation(fig, update, frames=n_chunks + 1, blit=False)
    return save_gif(anim, fig, "m14_training.gif", fps=4)


# ======================================================================================
# Module 15a
# ======================================================================================
def _m15a_fit():
    from solutions import m15a_flows as m
    from solutions.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob

    def compute():
        params = m.init_flow(key(1501), UPLIFT_DIM, 4, 32)
        params, trace = m.fit_flow_vi(uplift_centered_log_prob, params, key(1502), 12_000, lr=2e-3, n_samples=64)
        return dict(params=params, trace=trace)

    d = cached("m15a_flow", compute)
    return m, jax.tree_util.tree_map(jnp.asarray, d["params"])


def _m15a_layer_outputs(m, params, eps):
    masks = m.make_masks(eps.shape[1], len(params["layers"]))
    outs = [np.asarray(eps)]
    x = eps
    for layer, mask in zip(params["layers"], masks):
        x = jax.vmap(lambda z: m.coupling_forward(layer, mask, z)[0])(x)
        outs.append(np.asarray(x))
    x = x * jnp.exp(params["act_log_scale"]) + params["act_shift"]
    outs.append(np.asarray(x))
    return outs


@figure("m15a_flow_layers.png")
def m15a_flow_layers():
    m, params = _m15a_fit()
    eps = jax.random.normal(key(1510), (3000, 10))
    outs = _m15a_layer_outputs(m, params, eps)
    titles = ["base N(0, I)", "after 1 coupling", "after 2 couplings", "after 4 couplings", "+ affine: output"]
    picks = [0, 1, 2, 4, 5]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    for ax, i, t in zip(axes, picks, titles):
        ax.scatter(outs[i][:, 2], outs[i][:, 1], s=2, alpha=0.4, color=ORANGE)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("theta_1")
    axes[0].set_ylabel("log tau")
    fig.suptitle("A RealNVP flow builds the funnel one coupling layer at a time", y=1.03)
    return save(fig, "m15a_flow_layers.png")


@figure("m15a_flow_vs_hmc.png")
def m15a_flow_vs_hmc():
    from solutions.m08_bbvi import MeanFieldGaussian, fit
    from solutions.m09_hmc import UPLIFT_DIM, uplift_centered_log_prob

    m, params = _m15a_fit()
    ref = cached("m15a_reference", lambda: m.hmc_reference(key(1504)))
    mf = cached("m15a_meanfield", lambda: fit(uplift_centered_log_prob, UPLIFT_DIM, key(1503), 4000, lr=0.01, n_samples=16)[0])
    mf = jax.tree_util.tree_map(jnp.asarray, mf)
    flow_s, _ = m.flow_sample(key(1511), params, 4000)
    mf_s = MeanFieldGaussian(UPLIFT_DIM).sample(key(1512), mf, 4000)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, s, name, col in [(axes[0], np.asarray(ref), "HMC reference", INK), (axes[1], np.asarray(flow_s), "RealNVP flow", ORANGE), (axes[2], np.asarray(mf_s), "mean-field Gaussian", BLUE)]:
        ax.scatter(s[:, 2], s[:, 1], s=3, alpha=0.3, color=col)
        ax.set_title(f"{name}\nmass below log tau = -1: {float(m.tail_mass(jnp.asarray(s))):.2f}", fontsize=9.5)
        ax.set_xlabel("theta_1")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 3)
    axes[0].set_ylabel("log tau")
    return save(fig, "m15a_flow_vs_hmc.png")


@figure("m15a_flow_morph.gif")
def m15a_flow_morph_gif():
    m, params = _m15a_fit()
    eps = jax.random.normal(key(1510), (2500, 10))
    outs = _m15a_layer_outputs(m, params, eps)
    names = ["base N(0, I)", "coupling 1", "coupling 2", "coupling 3", "coupling 4", "affine output"]
    per = 8
    frames = []
    for i in range(len(outs) - 1):
        for k in range(per):
            a = k / per
            frames.append(((1 - a) * outs[i] + a * outs[i + 1], f"{names[i]} -> {names[i + 1]}"))
    for _ in range(10):
        frames.append((outs[-1], names[-1]))
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    sc = ax.scatter(frames[0][0][:, 2], frames[0][0][:, 1], s=3, alpha=0.4, color=ORANGE)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 4)
    ax.set_xlabel("theta_1")
    ax.set_ylabel("log tau")

    def update(i):
        pts, name = frames[i]
        sc.set_offsets(pts[:, [2, 1]])
        ax.set_title(f"Flow samples: {name}", fontsize=10)
        return (sc,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    return save_gif(anim, fig, "m15a_flow_morph.gif", fps=8)


# ======================================================================================
# Module 15b
# ======================================================================================
def _m15b_fit():
    from solutions import m15b_diffusion as m

    X = jnp.asarray(data.torsion_angles())
    d = cached("m15b_score_model", lambda: dict(zip(("params", "losses"), m.train_score_model(key(1520), X, n_steps=6000, batch_size=256, lr=2e-3, hidden=128))))
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    return m, X, params, np.asarray(d["losses"])


@figure("m15b_forward_noising.png")
def m15b_forward_noising():
    m, X, params, losses = _m15b_fit()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    keys = jax.random.split(key(1530), X.shape[0])
    for ax, t in zip(axes, [0.0, 0.2, 0.5, 1.0]):
        xt = jax.vmap(lambda x, k: m.sample_forward(k, x, jnp.asarray(t)))(X, keys)
        ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.5, color=ORANGE)
        ax.set_title(f"t = {t}: signal x {float(jnp.sqrt(m.alpha_bar(jnp.asarray(t)))):.3f}, noise sd {float(jnp.sqrt(1 - m.alpha_bar(jnp.asarray(t)))):.2f}", fontsize=8.5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("phi")
    axes[0].set_ylabel("psi")
    return save(fig, "m15b_forward_noising.png")


@figure("m15b_score_field.png")
def m15b_score_field():
    m, X, params, losses = _m15b_fit()
    score_fn = m.make_score_fn(params)
    g = jnp.linspace(-3.5, 3.5, 22)
    G1, G2 = jnp.meshgrid(g, g)
    pts = jnp.stack([G1.ravel(), G2.ravel()], 1)
    S = np.asarray(jax.vmap(lambda p: score_fn(p, jnp.asarray(0.1)))(pts))
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=2, color=GREY, alpha=0.4, label="data")
    ax.quiver(pts[:, 0], pts[:, 1], S[:, 0], S[:, 1], angles="xy", scale=250, color=ORANGE, width=0.004, label="learned score at t = 0.1")
    ax.set_xlabel("phi")
    ax.set_ylabel("psi")
    ax.set_aspect("equal")
    ax.set_title("The score points toward density: a vector field on data space")
    ax.legend(fontsize=8, loc="upper right")
    return save(fig, "m15b_score_field.png")


@figure("m15b_reverse_samples.png")
def m15b_reverse_samples():
    m, X, params, losses = _m15b_fit()
    score_fn = m.make_score_fn(params)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].scatter(X[:2000, 0], X[:2000, 1], s=2, alpha=0.5, color=INK)
    axes[0].set_title("data")
    for ax, n in zip(axes[1:], [50, 200, 1000]):
        xs = m.sample_reverse(key(1531), score_fn, 2000, n)
        ax.scatter(xs[:, 0], xs[:, 1], s=2, alpha=0.5, color=ORANGE)
        ax.set_title(f"{n} reverse steps")
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("phi")
    axes[0].set_ylabel("psi")
    return save(fig, "m15b_reverse_samples.png")


@figure("m15b_dsm_loss.png")
def m15b_dsm_loss():
    m, X, params, losses = _m15b_fit()
    sm = np.convolve(losses, np.ones(50) / 50, mode="valid")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(sm, color=ORANGE, label="DSM loss, 50-step average")
    ax.axhspan(0.36, 0.44, color=GREY, alpha=0.3, label="irreducible floor (variance of the conditional target)")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"The loss plateaus near {sm[-100:].mean():.2f}, not at zero")
    ax.legend(fontsize=8)
    return save(fig, "m15b_dsm_loss.png")


@figure("m15b_reverse_bayes.png")
def m15b_reverse_bayes():
    x = np.linspace(-4, 4, 2000)
    p_t = 0.6 * gauss_pdf(x, -1.5, 0.5) + 0.4 * gauss_pdf(x, 1.5, 0.5)
    b_dt = 0.3  # beta(t) * dt, exaggerated so the shift is visible
    x_t = 0.6
    fwd_mean = (1 - b_dt / 2) * x_t
    fwd = gauss_pdf(x, fwd_mean, np.sqrt(b_dt))
    x_next = fwd_mean  # the observed next point
    rev_unnorm = gauss_pdf(x_next, (1 - b_dt / 2) * x, np.sqrt(b_dt)) * p_t
    rev = rev_unnorm / np.trapezoid(rev_unnorm, x)
    rev_mean = np.trapezoid(x * rev, x)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.fill_between(x, p_t, color=GREY, alpha=0.4, label="marginal p_t(x)")
    ax.plot(x, fwd, color=BLUE, lw=1.8, label="forward kernel p(x_{t+dt} | x_t) from x_t")
    ax.plot(x, rev, color=ORANGE, lw=2.2, label="reverse kernel p(x_t | x_{t+dt}) = forward x p_t, normalised")
    ax.set_ylim(0, 1.25)
    for v, col, name, h, ha in [(x_t, BLUE, "x_t ", 1.1, "right"), (x_next, INK, "x_{t+dt} ", 0.95, "right"), (rev_mean, ORANGE, " E[x_t | x_{t+dt}]", 1.1, "left")]:
        ax.axvline(v, color=col, lw=1, ls="--")
        ax.text(v, h, name, ha=ha, va="bottom", fontsize=8.5, color=col)
    ax.annotate("", xy=(rev_mean, 0.84), xytext=(x_next, 0.84), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
    ax.text(2.6, 0.78, "shift toward density\nabout beta dt grad log p_t", ha="center", fontsize=8.5, color=ORANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("One reverse step is Bayes' rule: prior p_t, likelihood the forward kernel", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    return save(fig, "m15b_reverse_bayes.png")


@figure("m15b_diffusion.gif")
def m15b_diffusion_gif():
    m, X, params, losses = _m15b_fit()
    score_fn = m.make_score_fn(params)
    Xn = np.asarray(X[:1500])
    eps = np.random.default_rng(9).normal(size=Xn.shape)
    frames = []
    for t in np.linspace(0, 1, 24):
        ab = float(m.alpha_bar(jnp.asarray(t)))
        frames.append((np.sqrt(ab) * Xn + np.sqrt(1 - ab) * eps, f"forward noising, t = {t:.2f}"))
    # reverse SDE, recording every 9th of 432 steps
    n_steps = 432
    ts = np.linspace(1.0, m.T_EPS, n_steps + 1)
    dt = (1.0 - m.T_EPS) / n_steps
    drift = jax.jit(lambda x, t: jax.vmap(lambda xi: m.reverse_drift(score_fn, xi, t))(x))
    k = key(1540)
    x = jnp.asarray(np.random.default_rng(10).normal(size=Xn.shape))
    for i in range(n_steps):
        k, sub = jax.random.split(k)
        t = jnp.asarray(ts[i])
        noise = 0.0 if i == n_steps - 1 else jnp.sqrt(m.beta(t) * dt) * jax.random.normal(sub, x.shape)
        x = x - drift(x, t) * dt + noise
        if i % 9 == 8:
            frames.append((np.asarray(x), f"reverse SDE, t = {float(ts[i + 1]):.2f}"))
    fig, ax = plt.subplots(figsize=(5, 4.8), dpi=90)
    sc = ax.scatter(frames[0][0][:, 0], frames[0][0][:, 1], s=3, alpha=0.5, color=ORANGE)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("phi")
    ax.set_ylabel("psi")

    def update(i):
        pts, name = frames[i]
        sc.set_offsets(pts)
        sc.set_color(ORANGE if name.startswith("forward") else BLUE)
        ax.set_title(f"{name}  (frame {i + 1}/{len(frames)})", fontsize=10)
        return (sc,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    return save_gif(anim, fig, "m15b_diffusion.gif", fps=8)


# ======================================================================================
# Module 15c
# ======================================================================================
def _draw_dag(ax, nodes, edges, title, boxed=(), dashed=(), xlim=(-0.2, 3.2)):
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.3, 2.5)
    ax.axis("off")
    for name, (x, y) in nodes.items():
        shape = patches.Rectangle((x - 0.22, y - 0.22), 0.44, 0.44, fc="#fff7ed", ec=INK, lw=1.4) if name in boxed else patches.Circle((x, y), 0.24, fc="#fff7ed" if not name.startswith("U") else "#f3f4f6", ec=INK, lw=1.4)
        ax.add_patch(shape)
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5)
    for a, b in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        d = np.array([x1 - x0, y1 - y0])
        d = d / np.linalg.norm(d)
        ax.annotate("", xy=(x1 - 0.27 * d[0], y1 - 0.27 * d[1]), xytext=(x0 + 0.27 * d[0], y0 + 0.27 * d[1]), arrowprops=dict(arrowstyle="->", color=ORANGE if (a, b) in dashed else INK, lw=1.4, ls="--" if (a, b) in dashed else "-"))
    ax.set_title(title, fontsize=9.5)


@figure("m15c_dags.png")
def m15c_dags():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    base = {"S": (1.5, 2.1), "A": (0.3, 1.0), "T": (1.5, 0.2), "C": (2.7, 1.0)}
    edges = [("S", "A"), ("S", "T"), ("S", "C"), ("A", "T"), ("T", "C")]
    _draw_dag(axes[0], base, edges, "(a) observational: S confounds A and C")
    _draw_dag(axes[1], base, [e for e in edges if e != ("S", "A")], "(b) under do(A): the arrow into A is cut", boxed=("A",))
    twin = {"U_A": (0.1, 2.2), "U_T": (0.1, 1.2), "U_C": (0.1, 0.2), "S": (1.1, 1.2), "A": (2.0, 2.2), "T": (2.8, 2.2), "C": (3.6, 2.2), "A'": (2.0, 0.2), "T'": (2.8, 0.2), "C'": (3.6, 0.2)}
    tedges = [("U_A", "A"), ("U_T", "T"), ("U_C", "C"), ("U_T", "T'"), ("U_C", "C'"), ("S", "A"), ("S", "T"), ("S", "C"), ("A", "T"), ("T", "C"), ("S", "T'"), ("S", "C'"), ("A'", "T'"), ("T'", "C'")]
    _draw_dag(axes[2], twin, tedges, "(c) twin network: factual row above, counterfactual row below,\nshared noise, do(A') cuts the arrows into A'", boxed=("A'",), xlim=(-0.2, 4.0))
    return save(fig, "m15c_dags.png")


@figure("m15c_regression_vs_causal.png")
def m15c_regression_vs_causal():
    d = data.ad_spend_observational()
    scm = data.AdSpendSCM()
    A, C, S = d["A"], d["C"], d["S"]
    slope = np.cov(A, C)[0, 1] / A.var()
    causal = scm.c_t * scm.b_a
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sc = ax.scatter(A, C, c=S, cmap="coolwarm", s=5, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="seasonality S (the confounder)")
    g = np.linspace(A.min(), A.max(), 2)
    ax.plot(g, C.mean() + slope * (g - A.mean()), color=INK, lw=2.2, label=f"regression of C on A: slope {slope:.2f}")
    ax.plot(g, C.mean() + causal * (g - A.mean()), color=ORANGE, lw=2.2, ls="--", label=f"causal effect dE[C | do(A)]/dA = {causal:.2f}")
    ax.set_xlabel("ad spend A")
    ax.set_ylabel("conversions C")
    ax.set_title("Observation is not intervention: high-season weeks lift both A and C")
    ax.legend(fontsize=8, loc="upper left")
    return save(fig, "m15c_regression_vs_causal.png")


@figure("m15c_counterfactual_posterior.png")
def m15c_counterfactual_posterior():
    from solutions import m15c_scm as m

    d = data.ad_spend_observational()
    scm = data.AdSpendSCM()
    obs = {"A": jnp.asarray(d["A"][0]), "T": jnp.asarray(d["T"][0])}
    full = {k: jnp.asarray(d[k][0]) for k in ("S", "A", "T", "C")}
    a_new = jnp.asarray(0.26)
    post = cached("m15c_twin_posterior", lambda: np.asarray(m.twin_posterior(scm, obs, a_new, key(1550))["C_cf"]))
    mean, sd = m.exact_counterfactual_C(scm, obs, a_new)
    exact_full = float(m.counterfactual_closed_form(scm, full, a_new))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(post, bins=50, density=True, color=ORANGE, alpha=0.6, label=f"twin-network HMC draws of C under do(A = 0.26), n = {len(post)}")
    g = np.linspace(float(mean) - 4 * float(sd), float(mean) + 4 * float(sd), 400)
    ax.plot(g, gauss_pdf(g, float(mean), float(sd)), color=INK, lw=2, label=f"closed form N({float(mean):.3f}, {float(sd):.3f}^2)")
    ax.axvline(exact_full, color=BLUE, lw=2, ls="--", label=f"if C had also been observed: {exact_full:.3f}, no uncertainty")
    ax.set_xlabel("counterfactual conversions C_{a'}")
    ax.set_ylabel("posterior density")
    ax.set_title("Week one: what would conversions have been at half the spend?")
    ax.legend(fontsize=7.5, loc="upper left")
    return save(fig, "m15c_counterfactual_posterior.png")


# ======================================================================================
# Data export for widgets
# ======================================================================================
@figure("data/torsion_sample.json")
def torsion_json():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    pts = data.torsion_angles()[:1500]
    path = OUT / "data" / "torsion_sample.json"
    path.write_text(json.dumps([[round(float(a), 4), round(float(b), 4)] for a, b in pts]))
    print("wrote", path.relative_to(ROOT))
    return path


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print("\n".join(FIGURES))
        return 0
    names = argv or list(FIGURES)
    t0 = time.time()
    for name in names:
        if name not in FIGURES:
            print(f"unknown figure {name!r}; use --list", file=sys.stderr)
            return 1
        t = time.time()
        FIGURES[name]()
        print(f"   {name}: {time.time() - t:.1f} s")
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
