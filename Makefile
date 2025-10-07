# Makefile — single source of truth for setup, tests, benches, and releases.
SHELL := /bin/bash
.DEFAULT_GOAL := help
.SHELLFLAGS := -eu -o pipefail -c

# Ensure pipx/poetry shims are visible in all recipe shells
PATH := $(HOME)/.local/bin:$(PATH)

# Prefer system Python for pipx bootstrap
SYS_PY := $(shell command -v python3 || command -v python || echo /usr/bin/python3)

# Prefer 3.12, but gracefully fall back
PY_BIN ?= $(shell command -v python3.12 || command -v python3 || command -v python || echo python)

# Poetry binary (resolved absolute path if available)
POETRY := $(shell command -v poetry || echo $(HOME)/.local/bin/poetry)

BENCH_BINS ?= tdigest_core codecs
BENCH_ARGS ?= --warm-up-time 0.04 --sample-size 30 --measurement-time 1.5

STYLE_OK    := $(shell tput setaf 2 2>/dev/null || echo)
STYLE_ERR   := $(shell tput setaf 1 2>/dev/null || echo)
STYLE_BOLD  := $(shell tput bold 2>/dev/null || echo)
STYLE_RESET := $(shell tput sgr0 2>/dev/null || echo)

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef


# --------------------------------------------------------------------------------
# Poetry bootstrap
# --------------------------------------------------------------------------------
.PHONY: ensure_poetry
ensure_poetry:
	@set -eu; \
	if command -v poetry >/dev/null 2>&1; then \
	  printf "$(STYLE_OK)✓ poetry$(STYLE_RESET)\n"; \
	else \
	  printf "$(STYLE_BOLD)==> poetry not found; installing with pipx$(STYLE_RESET)\n"; \
	  if ! command -v pipx >/dev/null 2>&1; then \
	    printf "pipx not found; installing to user site with %s\n" "$(SYS_PY)"; \
	    "$(SYS_PY)" -m pip install --user -q pipx; \
	  fi; \
	  export PATH="$$HOME/.local/bin:$$PATH"; \
	  pipx ensurepath >/dev/null 2>&1 || true; \
	  export PATH="$$HOME/.local/bin:$$PATH"; \
	  pipx install --quiet --python "$(SYS_PY)" "poetry>=1.8,<2.0"; \
	  if command -v poetry >/dev/null 2>&1; then \
	    printf "$(STYLE_OK)✓ poetry installed$(STYLE_RESET)\n"; \
	  else \
	    printf "$(STYLE_ERR)poetry still not on PATH (export $$HOME/.local/bin)$(STYLE_RESET)\n"; exit 1; \
	  fi; \
	fi


# --------------------------------------------------------------------------------
# Environment bootstrap
# --------------------------------------------------------------------------------
.PHONY: setup resetup nuke
setup: ensure_poetry ## Bootstrap environment: Poetry, venv, nextest, hooks, maturin smoke
	$(call banner,Checking required host tools)
	$(call need,poetry)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)

	$(call banner,Configuring Poetry to use in-project venv (.venv))
	$(POETRY) config virtualenvs.in-project true

	$(call banner,Creating virtualenv with $(PY_BIN) and installing deps)
	$(POETRY) env use $(PY_BIN) || { echo "$(STYLE_ERR)$(PY_BIN) not found$(STYLE_RESET)"; exit 1; }
	$(POETRY) install --with dev

	$(call banner,Installing Rust components and nextest)
	rustup component add clippy rustfmt
	cargo install cargo-nextest --locked || true

	$(call banner,Installing pre-commit hooks)
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit install --hook-type pre-push

	$(call banner,Building Rust extension (maturin develop) + import smoke)
	$(POETRY) run pip -q install -U maturin
	$(POETRY) run maturin develop --release
	$(POETRY) run python -c 'import sys, polars as pl; import polars_tdigest as pt; print("python", sys.version.split()[0], "| polars", pl.__version__, "| polars_tdigest", getattr(pt,"__version__","ok"))'

	$(call banner,Quick pytest smoke)
	@[ -d tests ] && $(POETRY) run pytest -q -k "not slow" || echo "no tests/ directory; skipping"

	@printf "$(STYLE_OK)Setup complete. Activate: source .venv/bin/activate$(STYLE_RESET)\n"

resetup: nuke setup ## Remove env + rebuild everything from scratch

nuke: ## Remove all build artefacts, venvs, caches
	$(call banner,Removing build artefacts and caches)
	rm -rf .venv target dist build .maturin .pytest_cache .ruff_cache
	@echo "$(STYLE_OK)Environment nuked$(STYLE_RESET)"


# --------------------------------------------------------------------------------
# Dev / build targets
# --------------------------------------------------------------------------------
.PHONY: fmt lint check test fulltest bench_quick maturin_smoke clean

fmt: ## Format Rust + Python
	cargo fmt --all
	$(POETRY) run ruff format .

lint: ## Lint (Rust + Python)
	cargo clippy --all-targets --all-features -- -D warnings
	$(POETRY) run ruff check . --fix

check: ## Compile check
	cargo check --all-targets --all-features


# --------------------------------------------------------------------------------
# Testing split
# --------------------------------------------------------------------------------
test: ## Quick tests (Rust + Python fast subset)
	@if command -v cargo-nextest >/dev/null; then \
	  echo ">>> using cargo-nextest (fast)"; cargo nextest run --all-features --no-fail-fast --jobs auto -E 'not slow'; \
	else \
	  echo ">>> cargo-nextest not installed; using cargo test"; cargo test --all-features -- --quiet --skip slow; \
	fi
	@[ -d tests ] && $(POETRY) run pytest -q -k "not slow" || true

fulltest: ## Full test suite (Rust + Python including slow ones)
	@if command -v cargo-nextest >/dev/null; then \
	  echo ">>> using cargo-nextest (full)"; cargo nextest run --all-features --no-fail-fast --jobs auto; \
	else \
	  cargo test --all-features -- --quiet; \
	fi
	@[ -d tests ] && $(POETRY) run pytest -q || true


# --------------------------------------------------------------------------------
# Benches / smoke
# --------------------------------------------------------------------------------
bench_quick: ## Quick benches
	@for b in $(BENCH_BINS); do \
	  echo ">>> criterion smoke: $$b"; \
	  cargo bench --quiet --bench $$b -- $(BENCH_ARGS) || exit 1; \
	done

maturin_smoke: ## Build wheel into env and import
	$(POETRY) run maturin develop --release
	$(POETRY) run python -c "import polars_tdigest; print('maturin smoke OK')"

clean: ## Clean target + dist
	cargo clean && rm -rf dist build


# --------------------------------------------------------------------------------
# Meta targets
# --------------------------------------------------------------------------------
.PHONY: precommit_fast prepush_full ci_local ship help

precommit_fast: fmt lint check test ## Fast checks (on commit)

prepush_full: precommit_fast maturin_smoke bench_quick ## Heavier gate (on push)

ci_local: ## Full local CI (used by make ship)
	$(call banner,Local CI starting)
	$(MAKE) prepush_full
	$(MAKE) fulltest
	$(call banner,Local CI passed)

ship: ## Full local CI, tag, build artifacts; PUBLISH=1 to publish
	$(MAKE) ci_local
	$(call banner,Building artifacts)
	$(POETRY) run maturin build --release -o dist
	cargo package --allow-dirty
	$(call banner,Tagging release)
	@V=$$( $(POETRY) version -s ); git tag -a "v$$V" -m "Release v$$V"
	@echo "Tagged v$$V"
	@if [ "$(PUBLISH)" = "1" ]; then \
	  $(call banner,Publishing (Poetry + Cargo)); \
	  $(POETRY) publish; cargo publish; \
	else \
	  echo "Dry run complete. To publish: PUBLISH=1 make ship"; \
	fi


# --------------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  sed -E 's/:.*##/: /' | column -s': ' -t
