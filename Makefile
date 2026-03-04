.PHONY: setup cv predict submit all test lint clean

PYTHON ?= python
CONFIG ?= configs/baseline_ridge.yaml

# ── Setup ──────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -e ".[dev]"

# ── Pipeline ───────────────────────────────────────────
cv:
	$(PYTHON) -m spectral_challenge.cli cv --config $(CONFIG)

predict:
	@test -n "$(RUN_DIR)" || (echo "ERROR: set RUN_DIR=runs/xxx" && exit 1)
	$(PYTHON) -m spectral_challenge.cli predict --config $(CONFIG) --run-dir $(RUN_DIR)

submit:
	@test -n "$(RUN_DIR)" || (echo "ERROR: set RUN_DIR=runs/xxx" && exit 1)
	$(PYTHON) -m spectral_challenge.cli submit --config $(CONFIG) --run-dir $(RUN_DIR)

# ── Convenience: CV → submit in one shot ───────────────
all: cv
	@echo "CV done. To submit, run:  make submit RUN_DIR=runs/<latest>"

# ── Quality ────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

format:
	$(PYTHON) -m ruff check --fix src/ tests/
	$(PYTHON) -m ruff format src/ tests/

# ── Housekeeping ───────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build
