# ==============================================================================
# Makefile — polars-tdigest (Rust core + CLI, Python extension, Java/JNI)
# ==============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:
MAKEFLAGS += --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help

# ----------------------------------------------------------------------
# Pretty output helpers
# ----------------------------------------------------------------------
STYLE_OK    := $(shell tput setaf 2 2>/dev/null || printf '\033[32m')
STYLE_ERR   := $(shell tput setaf 1 2>/dev/null || printf '\033[31m')
STYLE_BOLD  := $(shell tput bold 2>/dev/null     || printf '\033[1m')
STYLE_RESET := $(shell tput sgr0 2>/dev/null     || printf '\033[0m')

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef

# ----------------------------------------------------------------------
# Tools & paths
# ----------------------------------------------------------------------
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

CARGO   ?= cargo
POETRY  ?= poetry
JAVAC   ?= javac
JAVA    ?= java
PYTHON  ?= python3

JAVA_SRC := src-java
LIB_DIR  := target/release
CP       := $(JAVA_SRC)
LIBPATH  := $(LIB_DIR)

JAVA_CLASSES := \
  $(JAVA_SRC)/gr/tdigest_rs/TDigestNative.class \
  $(JAVA_SRC)/gr/tdigest_rs/TDigest.class \
  $(JAVA_SRC)/TestRun.class

BENCH_BINS ?= cdf_quantile codecs tdigest
BENCH_ARGS ?= --warm-up-time 0.04 --sample-size 30 --measurement-time 1.5

# CLI binary name (matches Cargo [[bin]] name)
CLI_BIN ?= tdigest_cli
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)
CARGO_FEATURES ?=

# ==============================================================================
# HELP (grouped)
# ==============================================================================
.PHONY: help
help:
	@printf "\n$(STYLE_BOLD)Core$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "setup"        "Bootstrap rustup + Poetry env"
	@printf "  %-18s %s\n" "build"        "Build Rust lib+CLI, Python ext, Java/JNI"
	@printf "  %-18s %s\n" "test"         "Run Rust tests + CLI smoke + Python tests + Java smoke"
	@printf "  %-18s %s\n" "fmt"          "Format Rust + Python"
	@printf "  %-18s %s\n" "lint"         "Clippy + Ruff"
	@printf "  %-18s %s\n" "clean"        "Clean Rust, Python, and Java artifacts"
	@printf "  %-18s %s\n" "help-me-run"  "Show mini examples for Rust CLI/lib, Python, Polars, Java"
	@printf "\n$(STYLE_BOLD)Rust$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "rust-build"   "cargo build --release (lib + all bins)"
	@printf "  %-18s %s\n" "rust-test"    "cargo test"
	@printf "  %-18s %s\n" "rust-cli-smoke" "Smoke: echo '0 1 2 3' | $(CLI_BIN) quantile -q 0.5 == 1.5"
	@printf "\n$(STYLE_BOLD)Python$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "py-build"     "maturin develop -r"
	@printf "  %-18s %s\n" "py-test"      "pytest"
	@printf "\n$(STYLE_BOLD)Java/JNI$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "java-build"   "Build JNI lib + compile Java shim"
	@printf "  %-18s %s\n" "java-demo"    "Run tiny demo (prints CDF & p50)"
	@printf "  %-18s %s\n" "java-test"    "Run demo and assert output"
	@printf "\n$(STYLE_BOLD)Occasional$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "bench"        "criterion benches"
	@printf "  %-18s %s\n" "wheel"        "Build release wheel (abi3)"

# ==============================================================================
# Setup
# ==============================================================================
.PHONY: setup
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup); $(call need,cargo); $(call need,git); $(call need,poetry)
	$(POETRY) --version
	$(call banner,Create Poetry venv + install deps)
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) env use 3.12 || true
	$(POETRY) install --with dev -q
	$(POETRY) run pip -q install -U maturin
	$(call banner,Quick Python import smoke)
	$(POETRY) run python -c 'import polars as pl, polars_tdigest as pt, sys; print("python", sys.version.split()[0], "| polars", pl.__version__, "| polars_tdigest ok")'

# ==============================================================================
# Core dev loop (everything)
# ==============================================================================
.PHONY: build fmt lint clean test help-me-run

build:
	$(call banner,Build: Rust lib + CLI)
	$(MAKE) rust-build
	$(call banner,Build: Python extension)
	$(MAKE) py-build
	$(call banner,Build: Java/JNI)
	$(MAKE) java-build
	@printf "$(STYLE_OK)✓ all components built$(STYLE_RESET)\n"

fmt:
	$(CARGO) fmt --all
	$(POETRY) run ruff format .

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	$(POETRY) run ruff check .

clean:
	$(call banner,Clean: Rust target/)
	$(CARGO) clean
	$(call banner,Clean: Java .class files)
	@find $(JAVA_SRC) -name "*.class" -delete
	$(call banner,Clean: Python build artfacts)
	@rm -f polars_tdigest/*.so
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@printf "$(STYLE_OK)✓ cleaned Rust, Python, and Java artifacts$(STYLE_RESET)\n"

test: rust-test rust-cli-smoke py-test java-test
	@echo "✅ all tests passed"

help-me-run:
	@printf "\n$(STYLE_BOLD)How to run things$(STYLE_RESET)\n"
	@printf "\n$(STYLE_BOLD)Rust — CLI$(STYLE_RESET)\n"
	@printf "  cargo build --release --bin $(CLI_BIN)\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) quantile -q 0.5  # -> 1.5\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) --probes '0,1.5,3' cdf\n"
	@printf "\n$(STYLE_BOLD)Rust — as a library$(STYLE_RESET)\n"
	@printf "  // Create a small bin (src/bin/try.rs) using polars_tdigest::tdigest::TDigest\n"
	@printf "  // then: cargo run --release --bin try\n"
	@printf "  // skeleton:\n"
	@printf "  // use polars_tdigest::tdigest::{TDigest, ScaleFamily};\n"
	@printf "  // fn main(){ let xs = vec![0.0,1.0,2.0,3.0];\n"
	@printf "  //   let d = TDigest::new_with_size_and_scale(100, ScaleFamily::K2).merge_sorted(xs);\n"
	@printf "  //   println!(\"p50={}\", d.estimate_quantile(0.5)); }\n"
	@printf "\n$(STYLE_BOLD)Python — module$(STYLE_RESET)\n"
	@printf "  poetry run python -c \"import polars_tdigest as pt; import numpy as np;\\n"
	@printf "xs=[0,1,2,3]; d=pt.TDigest(xs, max_size=100); print(d.estimate_quantile(0.5))\"\n"
	@printf "\n$(STYLE_BOLD)Polars — plugin$(STYLE_RESET)\n"
	@printf "  poetry run python -c \"import polars as pl, polars_tdigest as ptd;\\n"
	@printf "df=pl.DataFrame({'x':[0.0,1.0,2.0,3.0]});\\n"
	@printf "df=df.select(ptd.tdigest('x',100));\\n"
	@printf "print(df.select(ptd.estimate_cdf('x',[0.0,1.5,3.0])))\"\n"
	@printf "\n$(STYLE_BOLD)Java — JNI demo$(STYLE_RESET)\n"
	@printf "  make java-demo    # prints the CDF triplet and p50=1.5 line\n"

# ==============================================================================
# Rust
# ==============================================================================
.PHONY: rust-build rust-test rust-cli-smoke

rust-build:
	# Build all Rust artifacts (lib + all bins) in release
	$(CARGO) build --release

rust-test:
	$(CARGO) test -- --quiet

# Smoke: echo "0 1 2 3" → quantile(0.5) == 1.5
Q ?= 0.5
$(CLI_PATH):
	$(CARGO) build --release --bin $(CLI_BIN) --features "$(CARGO_FEATURES)"

rust-cli-smoke: $(CLI_PATH)
	@set -eu; \
	OUT="$$( echo '0 1 2 3' | '$(CLI_PATH)' quantile -q $(Q) )"; \
	echo "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ rust_cli_smoke passed"

# ==============================================================================
# Python
# ==============================================================================
.PHONY: py-build py-test
py-build:
	$(POETRY) run maturin develop -r

py-test:
	$(POETRY) run pytest -q

# ==============================================================================
# Java / JNI
# ==============================================================================
.PHONY: java-build java-demo java-test

java-build:
	$(call need,javac)
	$(call need,java)
	$(CARGO) build --release --no-default-features --features java --lib
	cd $(JAVA_SRC) && $(JAVAC) -d . gr/tdigest_rs/TDigestNative.java gr/tdigest_rs/TDigest.java TestRun.java


# Human-friendly demo
java-demo: java-build
	$(JAVA) -Djava.library.path=$(LIBPATH) -cp $(CP) TestRun

# CI-friendly assertion
# Accepts either "p50=1.5" or "p50 = 1.5" and finds the array line anywhere.
java-test: java-build
	@set -e; \
	OUT="$$( $(JAVA) -Djava.library.path=$(LIBPATH) -cp $(CP) TestRun )"; \
	echo "$$OUT"; \
	ARR_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^\[[[:space:]0-9\.,]+\]$$' || true)"; \
	P50_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^p50[[:space:]]*=[[:space:]]*1\.5$$' || true)"; \
	[ "$$ARR_LINE" = "[0.125, 0.5, 0.875]" ] || { echo "❌ array mismatch"; exit 1; }; \
	[ -n "$$P50_LINE" ] || { echo "❌ p50 line missing"; exit 1; }; \
	echo "✅ java_test passed"

# ==============================================================================
# Occasional
# ==============================================================================
.PHONY: bench wheel

bench:
	for b in $(BENCH_BINS); do \
	  echo ">>> criterion: $$b"; \
	  cargo bench --quiet --bench $$b -- $(BENCH_ARGS) || exit 1; \
	done

wheel:
	$(POETRY) run maturin build --release --manylinux 2_28 --zig
