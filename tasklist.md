# Task list

Progress tracker for the build. Mark `[x]` when done and verified. Resume from the first
unchecked item. Conventions: a module is "done" only when its page, starter, solution,
tests and notebook exist, `pytest tests/mXX --solutions` passes, and
`scripts/check_starter_fails.py` shows every mXX test failing on the starter.

## Phase 0 — Plan
- [x] PLAN.md written
- [x] PLAN.md approved by user 2026-10-07 (request: varied datasets — tech, marketing, molecular biology)

## Phase 1 — Scaffold
- [x] pyproject.toml with pinned versions; `uv venv` + install verified on this machine
- [x] Makefile (setup, serve, build-site, test, test-solutions, check-starter)
- [x] mkdocs.yml (Material, MathJax, nav grouped by day, admonitions)
- [x] site/index.md, site/stylesheets/extra.css, site/javascripts/mathjax.js
- [x] tests/conftest.py (--solutions aliasing, x64), tests/_util.py
- [x] scripts/check_starter_fails.py
- [x] scripts/make_starter.py (generates workshop/ stubs from `# [mXX step N]` markers in solutions/)
- [x] workshop/__init__.py, solutions/__init__.py, workshop/data.py, solutions/data.py
- [x] README.md (setup in <10 min, make targets)
- [x] .gitignore
- [x] AGENTS.md (user request 2026-10-07; also: laptop browsers only, no mobile/desktop UI tuning)

## Phase 2 — Day 0 (then STOP for style review together with Day 1)
- [x] m00 Setup: page, tests/m00/test_env.py
- [x] m01 JAX warm-up: page, starter, solution, tests, notebook
- [x] m02 Diagnostic: page (pass criterion + review list), starter, solution, tests

## Phase 3 — Day 1 (then STOP for style review)
- [x] m03 Exponential-family core: page, starter, solution, tests, notebook
- [x] m04 KL and Bregman geometry: page, starter, solution, tests, notebook
- [x] m05 Conjugacy: page, starter, solution, tests, notebook
- [x] Full suite passes with --solutions; check-starter passes; mkdocs build --strict
- [x] Day 0 + Day 1 reviewed by user 2026-10-07 ("looks good. Build it all out"); Quarto considered, staying with MkDocs Material

## Phase 4 — Day 2
- [x] m06 ELBO and CAVI
- [x] m07 Gradient estimators (score-function vs reparameterisation, variance plot)
- [x] m08 Black-box VI engine
- [x] Full suite + check-starter + strict site build (Day 2 fork verified; m06 14 tests ~10 s, m07 10 tests ~31 s, m08 11 tests ~8 s)

## Phase 5 — Day 3
- [x] m09 MH and Hamiltonian dynamics
- [x] m10 Adaptation
- [x] m11 Diagnostics and HMC-vs-VI comparison
- [x] Full suite + check-starter + strict site build (Day 3 fork verified; m09 10 tests 3 s, m10 10 tests 5 s, m11 12 tests 11 s)

## Phase 6 — Day 4
- [x] m12 Traces and effect handlers
- [x] m13 Inference through the PPL
- [x] m14 VAE as amortised VI
- [x] Full suite + check-starter + strict site build (Day 4 fork verified; m12 17 tests 3.5 s, m13 8 tests 6.8 s, m14 10 tests 9.6 s)

## Phase 7 — Day 5
- [x] day5/index.md (track selection)
- [x] m15a Normalising flows (17 tests, 21 s)
- [x] m15b Score-based diffusion (13 tests, 22 s)
- [x] m15c SCMs and counterfactuals (16 tests, 7.5 s)
- [x] report-template.md
- [x] Full suite + check-starter + strict site build (Day 5: m15a 17, m15b 13, m15c 16 tests)

## Phase 8 — Finish
- [x] instructor/facilitator-guide.md, instructor/day-schedules.md (with measured test timings); included in site via pymdownx.snippets
- [x] site/reference/references.md, troubleshooting-index.md (all days merged)
- [x] Verify every reference is real; remove any uncertain URL (cite title/author only) — references.md carries only titles/authors/venues plus arXiv ids
- [x] `make serve`; walk every page in browser: rendering, math, links, nav — scripted crawl of all 30 nav URLs: 0 broken links, 0 MathJax errors, 0 unrendered formulas, 8 sections on every module page; visual spot checks of m09, m15c, troubleshooting index
- [x] Final: full suite 224 tests / 130 s against solutions; check-starter 220 failed as intended; 220 stubs; README re-checked

## Phase 9 — First-principles revision (user request 2026-10-08: "too advanced; keep breadth, explain concepts, first principles; engineers without Bayesian experience")
- [x] site/reference/glossary.qmd
- [x] Exemplar page: site/day1/m03-expfam-core.qmd rewritten (Why this module exists; Background subsections; worked numbers; What this computes per step)
- [x] site/index.qmd rewritten for the audience; sidebar entries for m02b and glossary
- [x] instructor guide + schedules, AGENTS.md, PLAN.md, README updated for m02b and longer days
- [x] Day 0 expanded (m00, m01, m02) + NEW m02b built (14 tests, 7 s; page 4.6k words)
- [x] Day 1 expanded (m04, m05) — 3.8k and 4.0k words
- [x] Day 2 expanded (m06–m08) — 4.0k, 3.5k, 3.3k words
- [x] Day 3 expanded (m09–m11) — 4.5k, 3.7k, 3.6k words
- [x] Day 4 expanded (m12–m14) — 3.6k, 3.5k, 3.3k words
- [x] Day 5 expanded (m15a–c, day5/index) — 4.7k, 4.7k, 4.9k words
- [x] Troubleshooting index rows for m02b; facilitator-guide test count for m02b
- [x] Full suite (238 tests, 141 s) + check-starter (234 fail) + make build-site (27 pages) + browser MathJax crawl

## Phase 10 — Understanding pass: figures, GIFs, widgets (user request 2026-10-08: "Do it all. Lots of figures, gifs and even three.js animations")
- [x] `workshop/jacobian_check.py` + `solutions/` copy + `tests/m00/test_jacobian_check.py`
- [x] Glossary: Jacobian entry (#jacobian) and notation-clash table (#notation)
- [x] m02b: Jacobian section made canonical (#sec-jacobian) with mass-conservation picture, checker snippet, shrinkage prediction prompt, figures, Metropolis GIF, RWM surface widget
- [x] m04: KL asymmetry prediction prompt + figures, bimodal fit figure + KL-direction widget, Bregman tangent figure + 3-D widget, conjugate as tangent intercept, Newton damping figure, natural gradients moved to optional Step 4, m08 claim corrected
- [x] m06: mean-field variance prediction prompt + ellipse figure, per-factor derivations moved into Steps 1–3, CAVI GIF and figures, ELBO-vs-K figure
- [x] m09: notation note, step-size prediction prompt, RWM-vs-HMC figure + GIF + HMC surface widget, involution derivation of the acceptance rule, leapfrog phase figure + GIF, acceptance-vs-step figure, funnel-in-the-prior figure + 3-D widget, Jacobian check callout
- [x] m15b: notation note, forward-noising figure + (x, y, t) widget, discrete-time Bayes derivation of the reverse SDE + figure, score field, DSM loss, reverse samples, diffusion GIF
- [x] Jacobian check callouts on m05, m08, m09, m13, m15a, m15c; prediction prompts on m07, m11, m15c
- [x] Figures embedded on m07, m08, m10, m11, m12, m13, m14, m15a, m15c; "Which expectation" paragraph in every Checkpoint m02b–m15c
- [x] `site/widgets/` (5 widgets, EMBED.md, _test.html); `_quarto.yml` resources; `make figures`; AGENTS/PLAN/README updated
- [x] `scripts/make_figures.py` produces every file referenced from the pages (40 PNG, 6 GIF, deterministic, 127 s from scratch); `_MANIFEST.md` written; m11 page numbers reconciled to the regenerated runs (153 divergences, ESS 59, R-hat 1.07)
- [x] Full suite (--solutions) passes; check-starter 234 fail as intended; `make build-site` OK (28 pages, all links/anchors)
- [x] PR #1 review fixes: check_starter_fails.py now fails on a missing directory, an empty collection or a non-1 pytest status; tests/conftest.py ignores WORKSHOP_IMPL (flag only); m13 `unconstrained_log_density` reshapes before the log-Jacobian (+ regression test, m13 now 9 tests); m10 `dual_averaging_init` starts log_step_avg at log(step_size0). Suite 243 pass, starter 235 fail
- [x] Browser walk of m02b, m04, m09, m11, m15b on the rendered site: all five widgets mount and animate in place, prediction callouts collapse, GIFs play, console free of errors on every page
- [x] Facilitator guide updated (figures and widgets on pages; prediction prompts; m04 Step 4 optional; m11 funnel on the page; Jacobian checker)

- [x] Two follow-up GIFs (user request 2026-10-08): m11_funnel_chains.gif (chains running, divergences accumulating) on m11; m14_training.gif (reconstructions and bound during training) on m14

- [x] PR #1 third review fixes: m01 `cumulative_logsumexp` no longer NaNs on leading -inf; m13 `predictive` draws observed sites at the observation's shape; `data.enzyme_kinetics` rejects n not a positive multiple of 8. Regression tests added. Fourth review pass: predictive broadcasts size-1 batch dimensions, docstrings state the new requirements, two troubleshooting rows, broader data-helper tests (m00 15, m01 19, m13 11 tests; suite 253)

## Phase 11 — PR #1 deferred items (user request 2026-10-08)
- [x] Speed: m09 `leapfrog` carries the gradient (L + 1 evaluations, not 2L; bit-identical results, churn `run_chains` 2.1 s -> 1.15 s) + count test via `jax.debug.callback`. Caching compiled `run_chains` programs (static `log_prob`) was tried and reverted in review: the jit cache kept ~19 MB per new target function alive (1.1 GB after 40 calls vs a 400 MB plateau); it still compiles once per call
- [x] Duplication: m11 uses m07's `churn_log_joint` and `laplace` (copies deleted); m15a `init_conditioner` and m15b score net built on m14 `init_mlp` / `mlp(..., activation)`; Day 5 prerequisites name m14 Step 1
- [x] `make_starter.py`: function extents from `ast`, stray markers are an error, dead header branch removed; output byte-identical on all 20 modules
- [x] Page template: Day 5 Overviews open with "Why this module exists.", m00/m02 steps with "What this computes."; `check_site.py` now enforces both openers and the step-heading form
- [x] Figures regenerated from an empty cache (only m11_hmc_vs_vi, m15a_*, m15b_* changed); m11, m15a, m15b worked-example numbers reconciled to the current code (m15a and m15b were already stale at db3d57e)
- [x] Suite 254 pass with --solutions (144 s); check-starter 239 fail; `make build-site` OK (28 pages)

## Notes for resumption
- Starter files are GENERATED: edit solutions/, then `python scripts/make_starter.py <module>`.
- site/reference/troubleshooting-index.md and site/reference/references.md exist; extend per day.
- mkdocs.yml nav currently lists Day 0, Day 1 and Reference only; add each day's pages when built.
- Full solutions suite (m00–m15c): 224 tests, ~130 s on this laptop.
- 2026-10-07: user asked for Quarto after the full build; site converted from MkDocs Material to a Quarto website (site/_quarto.yml, *.qmd, callouts, scripts/check_site.py replaces `mkdocs build --strict`).
- Working dir: /Users/ravikalia/Code/github.com/bayesian_inference (not a git repo)
- Python 3.13.2, `uv` at /opt/homebrew/bin/uv
- Pins chosen 2026-10-07: jax 0.11.2, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
  pytest 9.1.1, mkdocs-material 9.7.7 (verify at Phase 1; adjust if wheels missing)

## Phase 11 — Publish (user request 2026-10-08: "do the quarto site and host it on github pages; take inspiration from my other workshops")
- [x] Repo made public (user decision; Pages is unavailable on private repos on this plan); Pages source set to GitHub Actions
- [x] LICENSE: text CC BY 4.0, code MIT (as nlp-llms and tensors-workshop); footer says so
- [x] Theme after nlp-llms: `site/custom.scss` tokens + rules, `site/custom-dark.scss` palette, light/dark toggle following the OS on first visit, vendored Inter + Source Serif 4, navy navbar with GitHub icon, footer, Previous/Next cards
- [x] Landing page: hero, six day cards, module path, outcomes, step rhythm; existing reviewed prose kept
- [x] Fixed six references broken by a wrapped line starting with a year (parsed as an ordered list); `check_site.py` now fails on `<ol start="NNN">`
- [x] `.github/workflows/publish.yml` (test, render, deploy); README, AGENTS.md updated
- [ ] First deploy green and the live site checked in a browser

