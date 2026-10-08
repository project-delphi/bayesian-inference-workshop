# AGENTS.md

Guidance for AI agents and humans working in this repository.

## What this is
A five-day self-paced workshop, "Bayesian Inference from Scratch", in the style of
workshops.aws. Participants edit `workshop/`; `solutions/` holds reference
implementations; `tests/` grades both; `site/` is the Quarto website source.
See `PLAN.md` for the module list and `tasklist.md` for build progress.

## Resume protocol
1. Read `tasklist.md`. Resume at the first unchecked item.
2. Activate the environment: `source .venv/bin/activate` (created by `make setup`).
3. Before marking a module done run all three:
   `pytest tests/mXX --solutions`, `python scripts/check_starter_fails.py mXX`,
   `make build-site` (renders with Quarto and runs `scripts/check_site.py`).
   If the module's figures changed, also `make figures` (or
   `python scripts/make_figures.py <name>` for one figure).
4. Tick the box in `tasklist.md` in the same change.

## Hard rules
- JAX only. No Optax, Flax, NumPyro, TFP, BlackJAX, PyMC or Stan. Adam, MLPs, bijectors
  are hand-written.
- `workshop/` stubs must `raise NotImplementedError  # Module X, Step Y`. Signatures and
  docstrings in `workshop/` and `solutions/` must be identical.
- Both packages use relative imports only (`from .m03_expfam import ...`). The
  `--solutions` flag works by aliasing `sys.modules["workshop"]` to `solutions`.
- Every test outside `tests/m00` must fail against the starter. Every test must pass
  against the solutions. Fixed PRNG keys via `tests/_util.py`. `jax_enable_x64` is on.
- Datasets are generated in `workshop/data.py`; no downloads, no external data files.
- Provided helper modules (`data.py`, `jacobian_check.py`) are byte-identical in
  `workshop/` and `solutions/` and are tested in `tests/m00` (the only suite allowed to
  pass on the starter).
- References must be real. If a URL is uncertain, cite title and authors only.
- No emoji, no motivational filler on pages. Direct, rigorous tone.
- Audience: engineers with no Bayesian background. Every concept is built from first
  principles on the page where it is first used; nothing is assumed beyond calculus,
  linear algebra and probability. Module 2b is the conceptual anchor; the glossary
  (site/reference/glossary.qmd) defines terms once.

## Site
- Target: modern laptop browsers only. Do not spend effort on mobile or desktop-app
  layout tuning, responsive breakpoints or print styles.
- Pages are `.qmd` with a YAML `title:`; the sidebar text lives in `site/_quarto.yml`.
- Module pages follow the eight-part template exactly (Overview, Learning objectives,
  Background, Steps, Checkpoint, Challenge, Going deeper, Troubleshooting).
- Overview opens with "**Why this module exists.**". Background is split into `###`
  subsections: the problem in plain terms, definitions, line-by-line derivation (a
  sentence per equation), a worked example with numbers, why it matters later, common
  misconceptions. Each Step opens with "**What this computes.**" and includes a REPL
  sanity check with an expected value. Exemplar: site/day1/m03-expfam-core.qmd.
- Module ids are global; a module inserted between two existing ones takes a letter
  suffix (m02b) rather than renumbering everything.
- Callouts: `::: {.callout-tip title="Checkpoint"}`, `::: {.callout-note title="Challenge"}`,
  `::: {.callout-warning title="Troubleshooting"}`. Steps are `### Step N — title`.
- Math via MathJax with `$...$` and `$$...$$`; macros (`\E`, `\R`, `\KL`, `\dd`, `\tr`,
  `\diag`, `\ELBO`) are defined in `include-in-header` in `_quarto.yml`.
- Instructor pages include `instructor/*.md` with `{{< include ../../instructor/... >}}`.
- Pages show signatures and docstrings from `workshop/`, never solution bodies.
- Every Checkpoint callout ends with a short paragraph headed
  "**Which expectation did you just compute?**" naming the posterior (or $q$)
  expectations the module estimated. Module 2b's table "everything is an expectation"
  is the anchor this refers back to.
- Prediction prompts: a bold "**Predict before reading on.**" paragraph posing a
  question, followed by `::: {.callout-note collapse="true" title="Answer"}`. Use them
  before a surprising number or result (KL asymmetry, mean-field variance, funnel
  divergences, confounded slope), not for routine facts.
- Every page that changes variables carries a
  `::: {.callout-important title="Jacobian check"}` callout that links to
  `day0/m02b-foundations.qmd#sec-jacobian` and shows a three-line check with
  `workshop/jacobian_check.py` for that page's own transform. Do not re-derive the
  change-of-variables formula on later pages; Module 2b is its single home.
- Notation clashes between days ($q$ as variational density vs HMC position, two
  meanings of "score") are tabulated in `reference/glossary.qmd#notation`; a page that
  switches convention says so in one bold "A note on letters" paragraph at the top of
  Background.

## Figures, GIFs and widgets
- Static figures and GIFs live in `site/figures/` and are generated only by
  `scripts/make_figures.py` (`make figures`) from `solutions/`; never hand-edit or
  hand-save a figure. One function per figure, registered in `FIGURES`; fixed keys;
  the manifest `site/figures/_MANIFEST.md` lists every file with a one-line description.
  Style: `#c2410c` for the learner/approximation/sampler series, `#1f2937` for the
  exact/reference series, `#2563eb` for a second comparison, `#dc2626` only for
  failures and divergences. GIFs stay under about 3 MB.
- Embed with `![Full-sentence caption that states what to see.](../figures/name.png)`;
  the caption is the alt text and must say what the reader should notice, not just what
  is plotted. Each figure sits next to the paragraph whose claim it shows.
- Interactive widgets live in `site/widgets/` as plain ES modules (`mount(el, opts)`,
  options also as `data-*`); three.js core only, from cdnjs; no bundler, no other CDN.
  `site/widgets/_EMBED.md` documents each widget and its snippet. Embed as a
  `::: {.callout-note title="Interactive"}` paragraph saying what to do and what to look
  for, followed by a ```` ```{=html} ```` block with a `div.bi-widget` and a module
  script. `site/widgets/_test.html` is a dev page, not part of the site. `_quarto.yml`
  lists `widgets/**` and `figures/**` as project resources.

## Layout
```
workshop/   starter package      solutions/  reference package
tests/mXX/  test_stepN_*.py      site/       Quarto project (_quarto.yml, *.qmd)
notebooks/  per-module .py (jupytext) sources   instructor/  facilitator guide
scripts/    check_starter_fails.py, make_starter.py, make_figures.py, check_site.py
site/figures/   generated PNG/GIF (make figures)    site/widgets/  three.js / canvas widgets
```

## Commands
```
make setup            create .venv and install pinned deps
make test             pytest against workshop/
make test-solutions   pytest against solutions/
make check-starter    assert every test fails on the starter
make serve            quarto preview on http://localhost:8000
make build-site       quarto render site + scripts/check_site.py
make check-site       site checks on an existing render
make figures          regenerate site/figures from solutions/ (slow steps cache under site/figures/_cache)
```
