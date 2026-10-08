# Bayesian Inference from Scratch

A five-day self-paced workshop in modern approximate inference for engineers with no
Bayesian background, built in the style of an AWS Workshop: a static Quarto site of
modules with numbered hands-on steps. Every concept is derived from first principles on
the page where it is first used.
Participants build an exponential-family library, variational inference engines,
Hamiltonian Monte Carlo with adaptation and diagnostics, an effect-handler probabilistic
programming language, and a capstone (normalising flows, score-based diffusion, or
structural causal models), all in JAX with no probabilistic-programming framework.

## Setup (under 10 minutes)

Requirements: Python 3.11–3.13, [`uv`](https://docs.astral.sh/uv/), and the
[Quarto CLI](https://quarto.org/docs/get-started/) for the site (`brew install --cask quarto`
on macOS; tested with Quarto 1.6.40). CPU only is fine.

```bash
make setup                 # creates .venv and installs pinned dependencies
source .venv/bin/activate
pytest tests/m00           # environment check: 4 passed
make serve                 # site at http://127.0.0.1:8000
```

Without `uv`: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Create `.venv` and install pinned dependencies |
| `make serve` | Preview the site locally with live reload (`quarto preview`) |
| `make build-site` | Render the site and run `scripts/check_site.py` (fails on broken links, missing anchors or sections) |
| `make check-site` | Run the site checks on an existing render |
| `make test` | Run all tests against the starter code in `workshop/` |
| `make test-solutions` | Run all tests against the reference code in `solutions/` |
| `make check-starter` | Assert that every test fails against the starter |
| `make figures` | Regenerate every figure and GIF in `site/figures/` from `solutions/` (`python scripts/make_figures.py --list` to see them) |

Per module and step:

```bash
pytest tests/m03                # one module
pytest tests/m03 -k step2       # one step
pytest tests/m03 --solutions    # same tests, reference implementation
```

## Layout

```
site/        Quarto website source (`_quarto.yml`): one page per module, grouped by day
workshop/    Starter package. Participants edit this. Stubs raise NotImplementedError.
solutions/   Reference implementations with identical signatures. Instructors may withhold.
tests/       pytest suites, tests/mXX/test_stepN_*.py
notebooks/   Optional exploration notebooks (jupytext .py percent format)
instructor/  Facilitator guide and day schedules (also rendered in the site)
scripts/     check_starter_fails.py, make_starter.py, check_site.py
```

## Modules

| Day | Modules |
|---|---|
| 0 | m00 Setup, m01 JAX warm-up, m02 Diagnostic, m02b Bayesian inference from first principles |
| 1 | m03 Exponential-family core, m04 KL and Bregman geometry, m05 Conjugacy |
| 2 | m06 ELBO and CAVI, m07 Gradient estimators, m08 Black-box VI engine |
| 3 | m09 MH and Hamiltonian dynamics, m10 Adaptation, m11 Diagnostics and comparison |
| 4 | m12 Traces and effect handlers, m13 Inference through the PPL, m14 Amortised VI (VAE) |
| 5 | Capstone tracks: m15a Normalising flows, m15b Score-based diffusion, m15c Structural causal models |

## Notebooks

```bash
uv pip install -e ".[notebooks]" --python .venv/bin/python
jupytext --to ipynb notebooks/*.py
WORKSHOP_IMPL=solutions jupyter lab      # or omit the variable to use workshop/
```

## Dependencies

Pinned in `pyproject.toml`: jax 0.11.2, jaxlib 0.11.2, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2, pytest 9.1.1. The site needs the Quarto CLI (1.6.40). GPU:
`pip install -e ".[gpu]"`.
