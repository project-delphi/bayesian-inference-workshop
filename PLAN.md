# Bayesian Inference from Scratch — Build Plan

A five-day, 35-hour, self-paced workshop in the style of workshops.aws. Participants
build the machinery of modern approximate inference in JAX; every module ends with a
passing test suite for code the participant wrote.

## 1. Module list and time budgets

Module IDs are global (m00–m15) so test directories, starter files and site pages line
up one-to-one. Each day is ~7 h including breaks; Day 0 is ~3 h of pre-work.

| Day | ID | Module | Build | Time |
|---|---|---|---|---|
| 0 | m00 | Setup | Environment, `make test`, site served locally | 0.5 h |
| 0 | m01 | JAX warm-up | `grad`/`vmap`/`jit`/`lax.scan`, PRNG key discipline, pytrees; a stable `logsumexp`, a batched Gaussian log-density, a `scan`-based cumulative product with custom gradient check | 1.5 h |
| 0 | m02b | Bayesian inference from first principles | Model as a generative story; Bayes' rule from the product rule; the evidence integral and the grid method's cost; the workshop in miniature on the qPCR model: grid posterior, posterior predictive, Gaussian fit by KL (VI in miniature), Laplace, random-walk Metropolis (MCMC in miniature) | 3.0 h |
| 0 | m02 | Diagnostic | Auto-graded prerequisite check: Gaussian conditioning/marginalisation, change of variables with log-Jacobian, matrix calculus (∇ log det, ∇ trace), KL between Gaussians, Jensen, convexity of log-partition. Page states pass criterion directly and lists what to review on failure | 1.0 h |
| 1 | m03 | Exponential-family core | `ExpFam` protocol: `sufficient_stats`, `log_base_measure`, `log_partition(η)`, `log_prob`; mean params as `∇A`, Fisher as `∇²A` via autodiff; Bernoulli, Categorical, Gaussian (full covariance via natural params `(Σ⁻¹μ, −½Σ⁻¹)`), Gamma, Dirichlet. Tested against Monte Carlo moments and `scipy.stats` | 2.5 h |
| 1 | m04 | KL and Bregman geometry | Closed-form KL as the Bregman divergence of `A`; mean→natural inversion by convex conjugate (Newton on `∇A(η)=μ`); Fisher–Rao metric; tested against MC KL estimates for every family in m03 | 2.0 h |
| 1 | m05 | Conjugacy | Conjugate prior as an exponential family over `(η, −A(η))`; posterior update as addition of sufficient statistics; marginal likelihood and posterior predictive in closed form; Beta–Bernoulli, Dirichlet–Categorical, Gaussian–Gaussian, Normal–Gamma. Tested against quadrature | 2.5 h |
| 2 | m06 | ELBO and CAVI | Derive ELBO; mean-field Bayesian GMM (Dirichlet weights, Gaussian means with conjugate prior, known variance); CAVI updates from m05 natural-parameter arithmetic; ELBO must be monotone; recovers planted clusters | 2.5 h |
| 2 | m07 | Gradient estimators | Score-function (REINFORCE) with baseline/control variate vs reparameterisation; Bayesian logistic regression with mean-field Gaussian guide; variance measured over fixed keys and plotted vs number of samples | 2.5 h |
| 2 | m08 | Black-box VI engine | Unconstrained parameterisation with bijectors (`softplus`, `exp`, `sigmoid`) and their log-det Jacobians; Adam; data subsampling with likelihood rescaling (SVI); `fit(log_joint, guide, key, …)`; compared against Laplace approximation and m05 closed forms | 2.0 h |
| 3 | m09 | MH and Hamiltonian dynamics | Random-walk MH; Hamiltonian, leapfrog integrator (symplectic, reversible, volume-preserving — each tested); HMC transition kernel; energy-error tracking | 2.0 h |
| 3 | m10 | Adaptation | Dual-averaging step size (Hoffman & Gelman Alg. 5); diagonal mass matrix from warm-up variance with windowed schedule; jittered trajectory length; NUTS treated conceptually with a tree-doubling challenge | 2.5 h |
| 3 | m11 | Diagnostics and comparison | Split-R̂, ESS (FFT autocorrelation, Geyer truncation), divergence detection by energy-error threshold; multi-chain `vmap`ped sampler; HMC vs m08 VI on the same logistic-regression posterior (means, marginal variances, pairwise correlations) | 2.5 h |
| 4 | m12 | Traces and effect handlers | Messenger stack: `sample`, `param`, `trace`, `replay`, `condition`, `seed`, `substitute`, `block`; `log_density(model, trace)`; ~200 lines total | 2.5 h |
| 4 | m13 | Inference through the PPL | `AutoNormal` guide generated from a trace; `svi(model, guide)` using m08; `hmc(model)` using m10/m11 with automatic unconstraining via m08 bijectors; both run on a model written in the PPL | 2.0 h |
| 4 | m14 | Amortised VI: a VAE | Encoder/decoder MLPs as `param` sites; local latents per datum; reparameterised ELBO through the PPL; transcriptional-program dataset (64 binary genes, K latent programs, deterministic); negative ELBO must beat independent-Bernoulli baseline | 2.5 h |
| 5 | m15a | Capstone A: normalising flows | RealNVP affine coupling in JAX; forward/inverse/log-det tested for consistency; flow as variational family against the hierarchical-uplift funnel posterior; compare ELBO and HMC reference from m11; report | 7.0 h |
| 5 | m15b | Capstone B: score-based diffusion | VP-SDE forward process, denoising score matching objective, time-conditioned score MLP, Euler–Maruyama reverse sampler, probability-flow ODE with Hutchinson log-likelihood; Ramachandran-style (φ, ψ) torsion-angle data; report | 7.0 h |
| 5 | m15c | Capstone C: SCMs and counterfactuals | `do` handler on the m12 PPL; abduction–action–prediction via m13 inference on exogenous noise; twin-network construction; ad-spend SCM; tested against closed-form linear-Gaussian counterfactuals; report | 7.0 h |

Totals: Day 0 = 6 h, Days 1–4 about 8 h each including reading, Day 5 = 7 h.

Revision of 2026-10-08: the user judged the first version too advanced for engineers
without Bayesian background. Breadth kept; every module's Background rebuilt from first
principles with derivations, worked numbers and misconceptions; Module 2b added as the
conceptual anchor; glossary added; estimated times raised by 30–45 min per module.

Revision of 2026-10-08 (second): the user asked for the understanding of the hardest
concepts to be improved with figures, GIFs and three.js animations. Added: a figure
pipeline (`scripts/make_figures.py`, forty PNGs and eight GIFs in `site/figures/`),
five interactive widgets in `site/widgets/`, a provided `jacobian_check.py` with a
"Jacobian check" callout on every page that changes variables, prediction prompts with
collapsed answers before surprising results, a "Which expectation did you just compute?"
paragraph in every Checkpoint, a notation table in the glossary, a direct detailed-balance
derivation of the HMC acceptance rule (m09), a discrete-time Bayes derivation of the
reverse SDE (m15b), the convex conjugate explained as a tangent-line intercept and
natural gradients moved into the optional Step 4 (m04), and the per-factor CAVI
derivations moved from Background into Steps 1 to 3 (m06).

Each module page follows the fixed eight-part template (Overview, Learning objectives,
Background, Steps, Checkpoint, Challenge, Going deeper, Troubleshooting). Build vs
reading split is targeted at 60/40 by making Background one derivation only and putting
every other derivation inside a step that ends in code.

## 2. File tree

```
bayesian_inference/
├── README.md                     setup in <10 min; make targets
├── PLAN.md
├── Makefile                      setup | serve | test | test-solutions | check-starter | build-site
├── pyproject.toml                pinned deps; `uv` or pip; CPU jax
├── site/                         Quarto website (`_quarto.yml`; rendered HTML in site/_site, gitignored)
│   ├── index.md                  landing page: what you'll build, how to navigate, bar
│   ├── stylesheets/extra.css
│   ├── javascripts/mathjax.js
│   ├── day0/{m00-setup,m01-jax-warmup,m02-diagnostic}.md
│   ├── day1/{m03-expfam-core,m04-kl-bregman,m05-conjugacy}.md
│   ├── day2/{m06-cavi,m07-gradient-estimators,m08-bbvi-engine}.md
│   ├── day3/{m09-hmc-core,m10-adaptation,m11-diagnostics}.md
│   ├── day4/{m12-effect-handlers,m13-ppl-inference,m14-vae}.md
│   ├── day5/{index,m15a-flows,m15b-diffusion,m15c-scm,report-template}.md
│   └── reference/{references,troubleshooting-index}.md
├── workshop/                     starter package (participants edit this)
│   ├── __init__.py
│   ├── m01_jax.py … m14_vae.py, m15a_flows.py, m15b_diffusion.py, m15c_scm.py
│   └── data.py                   deterministic synthetic datasets (bars, two-moons, logistic data)
├── solutions/                    same module names, full implementations, relative imports
├── tests/
│   ├── conftest.py               --solutions flag aliases `workshop` → `solutions`
│   ├── m00/test_env.py           environment check (exempt from "must fail on starter")
│   ├── m01/ … m15c/              one file per step: test_step1_*.py …
│   └── _util.py                  shared tolerances, keys, MC helpers
├── notebooks/                    one exploration notebook per module (plots, diagnostics)
├── instructor/
│   ├── facilitator-guide.md      schedule, timing, sticking points, discussion prompts
│   └── day-schedules.md
└── scripts/
    ├── check_starter_fails.py    asserts every non-m00 test fails against workshop/
    └── make_notebooks.py         generates notebooks from .py sources (keeps them reviewable)
```

## 3. Test strategy

- **Import aliasing.** `tests/conftest.py` adds `--solutions`. When set, `pytest_configure`
  registers `sys.modules["workshop"] = solutions` before collection, so every test's
  `from workshop.m03_expfam import …` resolves to the reference implementation.
  Both packages use only relative imports internally so cross-module dependencies
  (m13 using m08 and m10) resolve inside the same package. Notebooks use
  `WORKSHOP_IMPL=solutions` for the same switch.
- **Step granularity.** Each step has its own test file `tests/mXX/test_stepN_<name>.py`
  so `pytest tests/m03 -k step2` runs exactly that step's checks. The module checkpoint is
  `pytest tests/m03`.
- **Starter must fail.** `scripts/check_starter_fails.py` runs the suite against
  `workshop/` with `-rA` and fails if any test outside `tests/m00` passes. Starter stubs
  `raise NotImplementedError  # Module X, Step Y`.
- **Determinism and tolerance.** All randomness via explicit `jax.random.key(seed)` with
  seeds fixed in `tests/_util.py`; `jax_enable_x64` on in `conftest.py`. Monte Carlo
  comparisons use N = 50k–200k samples and tolerances set at ~4 MC standard errors;
  closed-form comparisons use `rtol=1e-5`. Stochastic optimisation tests check
  invariants (ELBO monotone under CAVI, ELBO within a band of a known value, posterior
  mean within k posterior SDs of truth) rather than exact values.
- **Time budget.** Full solutions suite targets < 5 min on a laptop CPU; no single module
  suite > 60 s. Measured and recorded in the facilitator guide.
- **Properties, not snapshots.** Leapfrog: reversibility, volume preservation (Jacobian
  det = 1), energy error O(ε²). Bijectors: `inverse(forward(x)) = x`, log-det matches
  autodiff Jacobian. Flows: same. Handlers: `replay(trace)` reproduces log-density;
  `condition` zeroes out the sample's randomness.

## 4. Dependencies (pyproject.toml)

```
python >= 3.11, < 3.14
jax == 0.11.2, jaxlib == 0.11.2   (CPU wheels; GPU via optional extra `gpu`)
numpy == 2.5.3, scipy == 1.18.1, matplotlib == 3.11.2
pytest == 9.1.1
Quarto CLI 1.6.40 (separate install; not a Python package)
jupyter (optional extra `notebooks`)
```

Versions are the current PyPI releases as of 2026-10-07 and will be verified to install
and pass on this machine before Day 0 is reported done.

## 5. Site conventions

- Quarto website, docked sidebar grouped by day, cosmo theme with a small SCSS layer.
- MathJax 3 with `$…$` and `$$…$$`; shared macros via `include-in-header`.
- Callouts: tip for Checkpoint, note for Challenge, warning for Troubleshooting.
- Steps: `### Step N — title`, each ending in a fenced verification command and its
  expected output.
- Code shown on pages is always the *signature and docstring* from `workshop/`, never
  the solution body.

## 6. Decisions and assumptions

- JAX only; no Optax, Flax, NumPyro, TFP or BlackJAX. Adam and MLPs are written by hand
  (short) so nothing is hidden.
- NUTS is explained and its invariants tested conceptually; participants implement
  step-size and mass adaptation plus jittered trajectory lengths. Full tree-building is a
  Challenge, not a required step. (Writing a correct NUTS in the time budget is not
  realistic, and the user's brief says "conceptually".)
- Datasets are generated deterministically in `workshop/data.py` (no downloads), each
  themed on a real problem class so models have interpretable parameters:
  - m05: A/B conversion rates (Beta–Bernoulli); marketing-channel attribution counts
    (Dirichlet–Categorical); qPCR log-expression replicates (Normal–Gamma).
  - m06: 2-D single-cell embedding with planted cell types (GMM / CAVI).
  - m07, m08, m11, m13: SaaS churn prediction (Bayesian logistic regression; tenure,
    weekly usage, support tickets, plan tier). m08 subsampling uses a 50k-row ad
    click-through table.
  - m09–m11, m15a: hierarchical uplift across regional A/B tests (eight-schools
    structure, centred vs non-centred; produces the funnel geometry that exposes
    divergences).
  - m13: Michaelis–Menten enzyme kinetics as a non-linear regression written in the PPL.
  - m14: synthetic transcriptional-program dataset (64 genes, K latent programs,
    binary presence/absence; rendered as 8×8 grids) for the VAE.
  - m15b: Ramachandran-style (φ, ψ) backbone torsion angles as a 2-D multimodal density.
  - m15c: ad-spend → traffic → conversions SCM with a seasonality confounder;
    counterfactual "what if spend had been halved in week t".
- Day 5 tracks are each a full-day module; participants do one. Report template is a
  shared page with track-specific prompts.
- `instructor/` ships with the repo; `solutions/` is a sibling package that an instructor
  can delete or move without breaking tests against `workshop/`.

## 7. Build order

1. Day 0 + Day 1 complete (pages, starter, solutions, tests, notebooks), run everything,
   stop for style review.
2. Day 2, Day 3, Day 4, Day 5 one at a time; full suite after each.
3. Facilitator guide, README, final `make build-site` (render + link/anchor/section checks), serve locally and walk every
   page for rendering, math and links.
