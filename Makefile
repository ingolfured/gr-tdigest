# ==============================================================================
# Makefile — gr-tdigest (Rust core + Python extension + Java/JNI)
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
STYLE_CODE  := $(shell tput setaf 6 2>/dev/null || printf '\033[36m')

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null 2>&1 || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef

# ----------------------------------------------------------------------
# Project naming
# ----------------------------------------------------------------------
CRATE_NAME      ?= gr_tdigest
PROJECT_SLUG    ?= gr-tdigest
JAR_ARTIFACT    ?= $(PROJECT_SLUG)-java

# ----------------------------------------------------------------------
# Tools & paths
# ----------------------------------------------------------------------
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

CARGO  ?= cargo
UV     ?= uv
GRADLE := $(shell [ -x "$(JAVA_SRC)/gradlew" ] && echo "$(JAVA_SRC)/gradlew" || (command -v gradle || echo "gradle"))


# Force uv to use the repo-root venv
UV_ENV := UV_PROJECT_ENVIRONMENT=$(PWD)/.venv

# Python layout
PY_DIR          := bindings/python
PY_PKG_DIR      := $(PY_DIR)/gr_tdigest
PY_TESTS_DIR    := $(PY_DIR)/tests

# Java layout
JAVA_SRC        := bindings/java
JAVA_BUILD_REL  := build
JAVA_LIBS_REL   := $(JAVA_BUILD_REL)/libs

# Build profile (dev by default). Release only in release-* targets.
PROFILE ?= dev
ifeq ($(PROFILE),release)
  CARGO_PROFILE_FLAG := --release
  CARGO_DIR := release
else
  CARGO_PROFILE_FLAG :=
  CARGO_DIR := debug
endif
LIB_DIR  := target/$(CARGO_DIR)

# Version (from Cargo.toml)
VER := $(shell sed -n 's/^version\s*=\s*"\(.*\)"/\1/p' Cargo.toml | head -1)

# Platform detection → shared library basename
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M := $(shell uname -m)
ifeq ($(findstring linux,$(UNAME_S)),linux)
  PLAT := linux
  LIBBASENAME := lib$(CRATE_NAME).so
else ifeq ($(findstring darwin,$(UNAME_S)),darwin)
  PLAT := macos
  LIBBASENAME := lib$(CRATE_NAME).dylib
else ifeq ($(findstring mingw,$(UNAME_S))$(findstring msys,$(UNAME_S)),mingwmsys)
  PLAT := windows
  LIBBASENAME := $(CRATE_NAME).dll
else
  PLAT := linux
  LIBBASENAME := lib$(CRATE_NAME).so
endif

ifeq ($(UNAME_M),x86_64)
  ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
  ARCH := x86_64
else ifeq ($(UNAME_M),aarch64)
  ARCH := aarch64
else ifeq ($(UNAME_M),arm64)
  ARCH := aarch64
else ifneq (,$(filter i386 i686 x86,$(UNAME_M)))
  ARCH := x86
else
  ARCH := $(UNAME_M)
endif

# CLI binary (leave as-is if unchanged)
CLI_BIN  ?= tdigest
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)

# Python distributions
DIST      ?= dist
DIST_DEV  ?= dist-dev

# Canonical API JAR path (symlink updated by release-jar)
API_JAR := $(DIST)/$(JAR_ARTIFACT)-latest.jar

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
.PHONY: help setup clean \
        build build-rust build-python build-java \
        test rust-test py-test \
        lint \
        smoke-rust-cli smoke-wheel  \
        release-rust release-wheel release-jar release

# ==============================================================================
# Setup
# ==============================================================================
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)
	$(call need,uv)
	$(call need,gradle)
	$(UV) --version
	$(call banner,Create repo .venv and install Python deps (all groups))
	$(UV_ENV) $(UV) python install 3.12 || true
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) sync --all-groups)
	$(call banner,Quick Python import smoke)
	$(UV_ENV) $(UV) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"

# ==============================================================================
# Build (dev by default)
# ==============================================================================
build: build-rust build-python build-java
	@printf "$(STYLE_OK)✓ all components built (PROFILE=$(PROFILE))$(STYLE_RESET)\n"

build-rust:
	$(call banner,Build: Rust lib + CLI ($(PROFILE)))
	$(CARGO) build $(CARGO_PROFILE_FLAG)
	$(CARGO) build $(CARGO_PROFILE_FLAG) --bin $(CLI_BIN)

# Always run maturin from bindings/python where its pyproject.toml lives
build-python:
	$(call banner,Build: Python extension (maturin develop, dev))
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run maturin develop -F python)

# Use *relative* Gradle paths after cd to avoid double-prefix issues.
build-java:
	$(call banner,Build: Java via Gradle (clean + jar))
	@set -eu; \
	( cd "$(JAVA_SRC)" && \
	  $(GRADLE) --no-daemon clean jar && \
	  LATEST="$$(find "$(JAVA_LIBS_REL)" -maxdepth 1 -type f -name '$(JAR_ARTIFACT)-*.jar' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)" && \
	  [ -n "$$LATEST" ] || { echo "$(STYLE_ERR)✗ No JAR produced under $(JAVA_SRC)/$(JAVA_LIBS_REL)$(STYLE_RESET)"; exit 1; } && \
	  printf "$(STYLE_OK)✓ Built %s$(STYLE_RESET)\n" "$$LATEST" )

# ==============================================================================
# Tests
# ==============================================================================
test: rust-test py-test
	@echo "✅ all tests passed"

rust-test:
	$(CARGO) test -- --quiet

# Explicit path to bindings/python/tests
py-test: build-python
	$(UV_ENV) $(UV) run pytest -q $(PY_TESTS_DIR)

# ==============================================================================
# Lint (autofix where possible) + docs must be warning-free
# ==============================================================================
lint:
	$(call banner,Format Rust)
	$(CARGO) fmt --all

	$(call banner,Rust: clippy --fix (may modify files))
	$(CARGO) clippy --fix --allow-dirty --allow-staged

	$(call banner,Python: ruff check --fix + ruff format)
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run ruff check --fix .)
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run ruff format .)

	$(call banner,Python: mypy (auto-install types if missing))
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run python -m mypy gr_tdigest --install-types --non-interactive)

	$(call banner,Rust docs: deny warnings)
	@set -euo pipefail; \
	export RUSTDOCFLAGS="-D warnings -D rustdoc::private_intra_doc_links -D rustdoc::broken_intra_doc_links"; \
	$(CARGO) doc --no-deps

# ==============================================================================
# Clean (single target)
# ==============================================================================
clean:
	$(call banner,Clean: Rust target/ and cargo)
	rm -rf target/ || true
	$(CARGO) clean || true
	$(call banner,Clean: Gradle build outputs)
	@(cd "$(JAVA_SRC)" && $(GRADLE) --no-daemon clean) || true
	rm -rf "$(JAVA_SRC)/$(JAVA_BUILD_REL)" || true
	$(call banner,Clean: Python build artifacts)
	@find "$(PY_DIR)" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -f  "$(PY_PKG_DIR)"/*.so "$(PY_PKG_DIR)"/*.dylib "$(PY_PKG_DIR)"/*.pyd "$(PY_PKG_DIR)"/*.dll || true
	rm -rf "$(PY_DIR)/build/" "$(PY_DIR)"/*.egg-info/ || true
	$(call banner,Clean: Wheel dists)
	rm -rf "$(DIST)" "$(DIST_DEV)" || true
	@printf "$(STYLE_OK)✓ cleaned all artifacts$(STYLE_RESET)\n"

# ==============================================================================
# Smoke tests
# ==============================================================================
Q ?= 0.5

smoke-rust-cli:
	@set -eu; \
	BIN="target/$(CARGO_DIR)/$(CLI_BIN)"; \
	[ -x "$$BIN" ] || { echo "❌ missing CLI at $$BIN; build first (PROFILE=$(PROFILE))"; exit 1; }; \
	OUT="$$( echo '0 1 2 3' | "$$BIN" --stdin --cmd quantile --p $(Q) --no-header --output csv | cut -d, -f2 )"; \
	printf "CLI p%.3g -> %s\n" "$(Q)" "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ smoke-rust-cli ok (PROFILE=$(PROFILE))"

smoke-wheel:
	@set -eu; \
	LATEST="$$(ls -1t "$(DIST)"/*.whl "$(DIST_DEV)"/*.whl 2>/dev/null | head -1 || true)"; \
	[ -n "$$LATEST" ] || { echo "❌ no wheel found in $(DIST)/ or $(DIST_DEV)/"; exit 1; }; \
	TMP="$$(mktemp -d)"; \
	$(UV) venv --python 3.12 --seed "$$TMP"; \
	$(UV) pip install --python "$$TMP/bin/python" --upgrade pip; \
	$(UV) pip install --python "$$TMP/bin/python" 'polars>=1.34.0,<2.0.0' numpy; \
	$(UV) pip install --python "$$TMP/bin/python" --no-index --find-links "$$(dirname "$$LATEST")" gr-tdigest; \
	"$$TMP/bin/python" -c "import gr_tdigest as td; d=td.TDigest.from_array([0.0,1.0,2.0,3.0], max_size=100, scale='k2'); print('p50=', d.quantile(0.5)); print('cdf=', td.cdf); print('ok')"; \
	rm -rf "$$TMP"; \
	echo "✅ smoke-wheel ok (file=$$LATEST)"

# ==============================================================================
# Releases (release profile only)
# ==============================================================================
release-rust:
	$(call banner,Release: Rust CLI (release) + smoke)
	$(MAKE) build-rust PROFILE=release
	$(MAKE) smoke-rust-cli PROFILE=release
	@printf "• CLI binary : %s\n" "target/release/$(CLI_BIN)"

release-wheel:
	$(call banner,Release: Python PROD wheel (manylinux_2_28 + zig) + smoke)
	mkdir -p "$(DIST)"
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run maturin build -r -F python --compatibility manylinux_2_28 --zig -o "$(abspath $(DIST))")
	@printf "$(STYLE_OK)✓ wheel(s) in $(DIST)$(STYLE_RESET)\n"
	$(MAKE) -f $(CURDIR)/Makefile smoke-wheel


release-jar:
	$(call banner,Release: Java (Gradle) + publish to dist)
	@set -eu; \
	( cd "$(JAVA_SRC)" && \
	  $(GRADLE) --no-daemon clean smoke ); \
	mkdir -p "$(DIST)"; \
	LATEST="$(JAVA_SRC)/$(JAVA_LIBS_REL)/$(JAR_ARTIFACT)-$(VER).jar"; \
	[ -f "$$LATEST" ]; \
	cp -v "$$LATEST" "$(DIST)/"; \
	BASENAME="$$(basename "$$LATEST")"; \
	ln -sfn "$$BASENAME" "$(API_JAR)"; \
	echo "• Published: $(DIST)/$$BASENAME"; \
	echo "• Symlink : $(API_JAR) -> $$BASENAME"



release: release-rust release-wheel release-jar
	@printf "\n$(STYLE_BOLD)==> Release complete$(STYLE_RESET)\n"
	@printf "  CLI binary : %s\n" "target/release/$(CLI_BIN)"
	@printf "  Wheels dir : %s\n" "$(DIST)"
	@printf "  Java JAR   : %s (symlink)\n" "$(API_JAR)"
	@printf "$(STYLE_OK)✓ All release artifacts built & smoke-tested$(STYLE_RESET)\n"

help:
	@printf "\n$(STYLE_BOLD)Core (dev by default)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "setup"          "Install toolchains; sync Python deps (bindings/python) into .venv"
	@printf "  %-22s %s\n" "build"          "Build Rust lib+CLI (dev), Python ext (dev), Java via Gradle"
	@printf "  %-22s %s\n" "build-rust"     "Build Rust lib+CLI (dev)"
	@printf "  %-22s %s\n" "build-python"   "Build Python extension (maturin develop, dev)"
	@printf "  %-22s %s\n" "build-java"     "Build Java JAR with Gradle (clean+jar)"
	@printf "  %-22s %s\n" "test"           "Run tests: rust + python"
	@printf "  %-22s %s\n" "lint"           "Autofix (ruff/clippy/format) + mypy + rustdoc (deny warnings)"
	@printf "  %-22s %s\n" "clean"          "Remove ALL build artifacts (Rust/Gradle/Python)"
	@printf "\n$(STYLE_BOLD)Releases (release profile)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "release-rust"   "Build CLI (release) and smoke"
	@printf "  %-22s %s\n" "release-wheel"  "Build PROD wheel (release) inline and smoke"
	@printf "  %-22s %s\n" "release-jar"    "Gradle jar -> dist/ + latest symlink + smoke"
	@printf "  %-22s %s\n" "release"        "release-rust + release-wheel + release-jar"
