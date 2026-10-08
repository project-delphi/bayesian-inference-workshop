PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: setup serve build-site check-site test test-solutions check-starter figures clean

setup:
	uv venv .venv --python 3.13
	uv pip install -e . --python .venv/bin/python
	@command -v quarto >/dev/null || echo "Quarto CLI not found: install from https://quarto.org/docs/get-started/ (brew install --cask quarto)"
	@echo "Done. Activate with: source .venv/bin/activate"

serve:
	quarto preview site --port 8000 --no-browser

build-site:
	quarto render site
	$(PY) scripts/check_site.py

check-site:
	$(PY) scripts/check_site.py

figures:
	$(PY) scripts/make_figures.py

test:
	$(PYTEST) tests

test-solutions:
	$(PYTEST) tests --solutions

check-starter:
	$(PY) scripts/check_starter_fails.py

clean:
	rm -rf site/_site site/.quarto .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
