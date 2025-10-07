# ==============================================================================
# Makefile — polars-tdigest: single source of truth for setup, tests, benches.
# ==============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:
MAKEFLAGS += --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help

# Paths
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)
POETRY := poetry
VENV_DIR := .venv
POETRY_MIN_VERSION := 2.2.0
POETRY_BIN := $(HOME)/.local/bin/poetry
BENCH_BINS ?= tdigest_core codecs
BENCH_ARGS ?= --warm-up-time 0.04 --sample-size 30 --measurement-time 1.5

# Pretty output
STYLE_OK    := $(shell tput setaf 2 2>/dev/null || printf '\033[32m')
STYLE_ERR   := $(shell tput setaf 1 2>/dev/null || printf '\033[31m')
STYLE_BOLD  := $(shell tput bold 2>/dev/null     || printf '\033[1m')
STYLE_RESET := $(shell tput sgr0 2>/dev/null     || printf '\033[0m')

CARGO_TERM_PROGRESS_WHEN ?= always   # auto|always|never
CARGO_TERM_PROGRESS_WIDTH ?= 80      # adjust to your terminal width

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef

# ==============================================================================
# Rust toolchain check
# ==============================================================================
.PHONY: ensure_rustup
ensure_rustup:
	if command -v rustup >/dev/null 2>&1; then
	  printf "$(STYLE_OK)✓ rustup present: %s$(STYLE_RESET)\n" "$$(rustup --version | cut -d' ' -f1-2)"
	else
	  printf "\n$(STYLE_ERR)rustup is required$(STYLE_RESET)\n\n"
	  printf "Install instructions:\n"
	  printf "  Ubuntu/Debian: sudo apt install -y curl build-essential && curl https://sh.rustup.rs | sh -s -- -y\n"
	  printf "  Arch: sudo pacman -Sy --noconfirm --needed rustup base-devel && rustup default stable\n"
	  exit 1
	fi

# ==============================================================================
# Poetry 2.x via pipx
# ==============================================================================
.PHONY: ensure_poetry
ensure_poetry:
	if ! command -v pipx >/dev/null 2>&1; then
	  printf ">> installing pipx (user)\n"
	  if command -v python3 >/dev/null 2>&1; then python3 -m pip install --user -q pipx; \
	  elif command -v python >/dev/null 2>&1; then python -m pip install --user -q pipx; \
	  else printf "$(STYLE_ERR)ERROR: no python to bootstrap pipx$(STYLE_RESET)\n"; exit 1; fi
	  pipx ensurepath >/dev/null 2>&1 || true
	  export PATH="$$HOME/.local/bin:$$PATH"
	fi
	if ! command -v poetry >/dev/null 2>&1; then
	  printf ">> installing poetry with pipx\n"
	  pipx install -q poetry
	  export PATH="$$HOME/.local/bin:$$PATH"
	fi
	V="$$(poetry --version 2>/dev/null | sed -E 's/.*version[[:space:]]+([0-9.]+).*/\1/')"
	if [ -z "$$V" ] || [ "$$(printf '%s\n' "$$V" "$(POETRY_MIN_VERSION)" | sort -V | tail -n1)" != "$$V" ]; then
	  printf ">> upgrading poetry (have %s, need >= %s)\n" "$${V:-none}" "$(POETRY_MIN_VERSION)"
	  pipx install -q --force poetry
	  V="$$(poetry --version | sed -E 's/.*version[[:space:]]+([0-9.]+).*/\1/')"
	fi
	command -v poetry >/dev/null 2>&1 || { printf "$(STYLE_ERR)ERROR: poetry not on PATH$(STYLE_RESET)\n"; exit 1; }
	printf "$(STYLE_OK)✓ Poetry %s at %s$(STYLE_RESET)\n" "$$V" "$(POETRY_BIN)"

# ==============================================================================
# Poetry-managed Python 3.12
# ==============================================================================
.PHONY: ensure_python312
ensure_python312: ensure_poetry
	$(call banner,Ensuring Poetry-managed Python 3.12)
	if $(POETRY) python list | awk '{print $$1}' | grep -q '^3\.12'; then
	  printf "$(STYLE_OK)✓ Python 3.12 already installed (Poetry)$(STYLE_RESET)\n"
	else
	  printf ">> poetry python install 3.12\n"
	  $(POETRY) python install 3.12
	  touch .managed_py312
	fi
	$(POETRY) env use 3.12
	EXE="$$( $(POETRY) run python -c 'import sys, os; assert sys.version_info[:2]==(3,12); print(os.path.realpath(sys.executable))' )"
	printf "%s\n" "$$EXE" > .python312_path
	printf "$(STYLE_OK)✓ Python 3.12 at %s$(STYLE_RESET)\n" "$$EXE"

# ==============================================================================
# Environment bootstrap
# ==============================================================================
.PHONY: setup resetup nuke
setup: ensure_rustup ensure_python312 ## Full bootstrap
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)

	$(call banner,Configuring Poetry to use in-project venv ($(VENV_DIR)))
	$(POETRY) config virtualenvs.in-project true

	$(call banner,Creating virtualenv + installing deps)
	$(POETRY) env use 3.12
	$(POETRY) install --with dev -q
	printf "$(STYLE_OK)✓ Dependencies installed$(STYLE_RESET)\n"

	$(call banner,Installing pre-commit hooks)
	# pre-commit doesn't support -q; keep it clean manually
	if $(POETRY) run pre-commit install >/dev/null 2>&1; then
	  printf "$(STYLE_OK)✓ pre-commit hook installed$(STYLE_RESET)\n";
	else
	  printf "$(STYLE_ERR)✗ failed to install pre-commit hook$(STYLE_RESET)\n";
	fi
	if $(POETRY) run pre-commit install --hook-type pre-push >/dev/null 2>&1; then
	  printf "$(STYLE_OK)✓ pre-push hook installed$(STYLE_RESET)\n";
	else
	  printf "$(STYLE_ERR)✗ failed to install pre-push hook$(STYLE_RESET)\n";
	fi


	$(call banner,Building Rust extension (maturin develop) + import smoke)
	# Ensure maturin is available in the venv
	$(POETRY) run pip -q install -U maturin

	# Build Rust extension with live progress bar + timings report
	CARGO_TERM_PROGRESS_WHEN=$(CARGO_TERM_PROGRESS_WHEN) \
	CARGO_TERM_PROGRESS_WIDTH=$(CARGO_TERM_PROGRESS_WIDTH) \
	.venv/bin/python -m maturin develop -r

	# Verify import
	$(POETRY) run python -c "import sys,polars as pl; import polars_tdigest as pt; \
	print('python',sys.version.split()[0],'| polars',pl.__version__,'| polars_tdigest',getattr(pt,'__version__','ok'))" \
	|| { printf 'import smoke failed\n'; exit 1; }


	$(call banner,Quick pytest smoke)
	if [ -d tests ]; then $(POETRY) run pytest -q -k "not slow"; else echo "no tests/; skipping"; fi

	printf "$(STYLE_OK)Setup complete. Activate: source $(VENV_DIR)/bin/activate$(STYLE_RESET)\n"


resetup: nuke setup ## Remove env + rebuild everything

nuke: ## Remove venv, build artefacts, caches
	$(call banner,Removing build artefacts and caches)
	rm -rf $(VENV_DIR) target dist build .maturin .pytest_cache .ruff_cache .python312_path
	if [ -f .managed_py312 ]; then
	  printf "Removing Poetry-managed Python 3.12 (installed by this Makefile)\n"
	  $(POETRY) python remove 3.12 || true
	  rm -f .managed_py312
	fi
	printf "$(STYLE_OK)Environment nuked$(STYLE_RESET)\n"

# ==============================================================================
# Dev / build targets
# ==============================================================================
.PHONY: fmt lint check test fulltest bench_quick maturin_smoke precommit_fast prepush_full

fmt: ## Format Rust + Python
	cargo fmt --all
	$(POETRY) run ruff format .

lint: ## Lint (Rust + Python)
	cargo clippy --all-targets --all-features -- -D warnings
	$(POETRY) run ruff check . --fix

check:
	cargo check --all-targets --all-features

test:
	cargo test -- --quiet
	pytest

fulltest: ## Full test suite
	cargo test -- --quiet --include-ignored
	pytest

bench_quick: ## Quick criterion benches
	for b in $(BENCH_BINS); do
	  echo ">>> criterion smoke: $$b"
	  cargo bench --quiet --bench $$b -- $(BENCH_ARGS) || exit 1
	done

maturin_smoke: ## Build wheel into env and import
	$(POETRY) run maturin develop --release -q
	$(POETRY) run python -c "import polars_tdigest; print('maturin smoke OK')"

# Composite targets for pre-commit
precommit_fast: fmt lint check test ## pre-commit stage
	@echo "precommit_fast done."

prepush_full: fulltest maturin_smoke bench_quick ## pre-push stage
	@echo "prepush_full done."

# -------------------- Wheel building & testing --------------------
.PHONY: wheel wheel_manylinux wheel_allpy wheel_clean wheel_install_local wheel_check

wheel: ## musllinux_1_2 (musl) via container
	$(POETRY) run maturin build --release --manylinux 2_28 --zig

# Build for *all* local interpreters you have (only needed if NOT using abi3)
wheel_allpy: ## Build wheels for all Python versions found
	$(POETRY) run maturin build --release --manylinux 2_28 --zig -i python3.9 -i python3.10 -i python3.11 -i python3.12 -i python3.13

wheel_clean: ## Clean dist
	rm -rf target/wheels/* target/maturin/*

WHEEL_DIR ?= target/wheels
WHEEL_GLOB := $(WHEEL_DIR)/polars_tdigest-*.whl

wheel_install_local: ## pip install the freshest built wheel
	$(eval WHEEL := $(lastword $(sort $(wildcard $(WHEEL_GLOB)))))
	@test -n "$(WHEEL)" || { echo "No wheel found in $(WHEEL_DIR)"; exit 1; }
	python -m pip install -U --force-reinstall "$(WHEEL)"
	@echo "Installed: $(WHEEL)"

wheel_check: ## sanity: metadata & manylinux audit
	$(POETRY) run python -m pip install -U twine auditwheel || true
	$(POETRY) run twine check dist/*.whl || true
	# On Linux, audit the wheel (maturin usually does this already)
	@if command -v auditwheel >/dev/null 2>&1; then auditwheel show dist/*.whl || true; fi


help: ## Show this help
	@printf "\n$(STYLE_BOLD)Available targets$(STYLE_RESET)\n\n"
	@awk 'BEGIN {FS=":.*## "}; \
	     /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST) | sort
