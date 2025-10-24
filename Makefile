# ==============================================================================
# Makefile — tdigest-rs (Rust core + Python extension + Java/JNI)
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
	@command -v $(1) >/dev/null 2>&1 || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef

# ----------------------------------------------------------------------
# Tools & paths
# ----------------------------------------------------------------------
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

CARGO ?= cargo
UV     ?= uv
JAVAC  ?= javac
JAVA   ?= java
PYTHON ?= python3

JAVA_SRC := src-java
LIB_DIR  := target/release
CP_DEV   := target/java-classes

# Version (from Cargo.toml)
VER := $(shell sed -n 's/^version\s*=\s*"\(.*\)"/\1/p' Cargo.toml | head -1)

# Platform detection
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M := $(shell uname -m)

ifeq ($(findstring linux,$(UNAME_S)),linux)
  PLAT := linux
  LIBBASENAME := libtdigest_rs.so
else ifeq ($(findstring darwin,$(UNAME_S)),darwin)
  PLAT := macos
  LIBBASENAME := libtdigest_rs.dylib
else ifeq ($(findstring mingw,$(UNAME_S))$(findstring msys,$(UNAME_S)),mingwmsys)
  PLAT := windows
  LIBBASENAME := tdigest_rs.dll
else
  PLAT := linux
  LIBBASENAME := libtdigest_rs.so
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

# JAR output locations
JAVA_PKG_DIR   := target/java
API_JAR        := $(JAVA_PKG_DIR)/tdigest-rs-java-$(VER).jar
API_MANIFEST   := $(JAVA_PKG_DIR)/MANIFEST.MF
NATIVE_JAR_CUR := $(JAVA_PKG_DIR)/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar
STAGE          := target/jar-staging

# Prebuilt natives (optional)
NATIVE_LINUX_X86_64    ?=
NATIVE_MACOS_AARCH64   ?=
NATIVE_WINDOWS_X86_64  ?=

# CLI binary
CLI_BIN  ?= tdigest
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)

# ==============================================================================
# HELP
# ==============================================================================
.PHONY: help
help:
	@printf "\n$(STYLE_BOLD)Core$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "setup"        "Install toolchain + Python env"
	@printf "  %-18s %s\n" "build"        "Build Rust, Python extension, and compile Java"
	@printf "  %-18s %s\n" "test"         "Rust tests + CLI smoke + Python tests + Java demo (classes)"
	@printf "  %-18s %s\n" "fmt"          "Format Rust + Python"
	@printf "  %-18s %s\n" "lint"         "Clippy (Rust) + Ruff (Python)"
	@printf "  %-18s %s\n" "clean"        "Clean Rust, Python, and Java artifacts"
	@printf "  %-18s %s\n" "help-me-run"  "Examples: Java (JAR & classes), Rust CLI, Python, Polars plugin"
	@printf "\n$(STYLE_BOLD)Rust$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "rust-build"   "cargo build --release"
	@printf "  %-18s %s\n" "rust-test"    "cargo test"
	@printf "  %-18s %s\n" "rust-cli-smoke" "Smoke: echo '0 1 2 3' | tdigest-rs --stdin --cmd quantile --p 0.5 --no-header == 1.5"
	@printf "\n$(STYLE_BOLD)Python$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "py-build"     "maturin develop -r -F python"
	@printf "  %-18s %s\n" "py-test"      "pytest -q"
	@printf "\n$(STYLE_BOLD)Java/JNI$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "java-build"   "Build JNI lib + compile Java classes to target/java-classes"
	@printf "  %-18s %s\n" "java-test"    "Run Java demo from classes (uses -Djava.library.path)"
	@printf "\n$(STYLE_BOLD)Publish$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "wheel"        "Build Python wheel (abi3)"
	@printf "  %-18s %s\n" "jar"          "Build API JAR + current-platform native JAR"
	@printf "  %-18s %s\n" "jar-all"      "Build API JAR + Linux/macOS/Windows native JARs"

# ==============================================================================
# Setup
# ==============================================================================
.PHONY: setup
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)
	$(call need,uv)
	$(UV) --version

	$(call banner,Create .venv and install deps (all groups))
	$(UV) python install 3.12 || true
	$(UV) sync --all-groups

	$(call banner,Quick Python import smoke)
	$(UV) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"

# ==============================================================================
# Core dev loop
# ==============================================================================
.PHONY: build fmt lint clean test help-me-run

build:
	$(call banner,Build: Rust lib + CLI)
	$(MAKE) rust-build
	$(call banner,Build: Python extension)
	$(MAKE) py-build
	$(call banner,Compile: Java)
	$(MAKE) java-build
	@printf "$(STYLE_OK)✓ all components built$(STYLE_RESET)\n"

fmt:
	$(CARGO) fmt --all
	$(UV) run ruff format .

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	$(UV) run ruff check .

clean:
	$(call banner,Clean: Rust target/)
	$(CARGO) clean
	$(call banner,Clean: Java classes & jars)
	 rm -rf $(CP_DEV) $(JAVA_PKG_DIR) $(STAGE) || true
	 find src-java -type f \( -name '*.class' -o -name '*.java.bak' \) -delete || true
	$(call banner,Clean: Python artifacts)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -f tdigest_rs/*.so || true
	@printf "$(STYLE_OK)✓ cleaned Rust, Python, and Java artifacts$(STYLE_RESET)\n"

test: rust-test rust-cli-smoke py-test java-test
	@echo "✅ all tests passed"

help-me-run:
	@printf "\n$(STYLE_BOLD)Rust — CLI$(STYLE_RESET)\n"
	@printf "  cargo build --release --bin $(CLI_BIN)\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) --stdin --cmd quantile --p 0.5 --no-header   # -> 1.5\n"
	@printf "  echo '0 1.5 3' | target/release/$(CLI_BIN) --stdin --cmd cdf                          # -> CSV x,p rows\n"
	@printf "\n$(STYLE_BOLD)Python — module$(STYLE_RESET)\n"
	@printf "  uv run python - <<'PY'\n"
	@printf "import tdigest_rs as pt\n"
	@printf "values = [0.0, 1.0, 2.0, 3.0]\n"
	@printf "d = pt.TDigest.from_array(values, max_size=100, scale='k2')\n"
	@printf "print('p50 =', d.quantile(0.5))\n"
	@printf "print('cdf =', d.cdf([0.0, 1.5, 3.0]).tolist())\n"
	@printf "PY\n"

# ==============================================================================
# Rust
# ==============================================================================
.PHONY: rust-build rust-test rust-cli-smoke

rust-build:
	$(CARGO) build --release

rust-test:
	$(CARGO) test -- --quiet

Q ?= 0.5
$(CLI_PATH):
	$(CARGO) build --release --bin $(CLI_BIN)

rust-cli-smoke: $(CLI_PATH)
	@set -eu; \
	OUT="$$( echo '0 1 2 3' \
	  | '$(CLI_PATH)' --stdin --cmd quantile --p $(Q) --no-header --output csv \
	  | cut -d, -f2 )"; \
	echo "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ rust_cli_smoke passed"

# ==============================================================================
# Python
# ==============================================================================
py-build:
	$(UV) run maturin develop -r -F python

py-test: py-build
	$(UV) run pytest -q

# ==============================================================================
# Java / JNI (classes build + smoke)
# ==============================================================================
.PHONY: java-build java-test

java-build:
	$(call need,javac)
	$(call need,java)
	$(CARGO) build --release --no-default-features --features java --lib
	rm -rf $(CP_DEV)
	mkdir -p $(CP_DEV)
	javac -d $(CP_DEV) $(shell find $(JAVA_SRC) -name '*.java')

java-test: java-build
	@set -e; \
	OUT="$$( $(JAVA) -Djava.library.path=$(LIB_DIR) -cp $(CP_DEV) TestRun )"; \
	echo "$$OUT"; \
	ARR_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^\[[[:space:]0-9\.,]+\]$$' || true)"; \
	P50_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^p50[[:space:]]*=[[:space:]]*1\.5$$' || true)"; \
	[ "$$ARR_LINE" = "[0.125, 0.5, 0.875]" ] || { echo "❌ array mismatch (classes)"; exit 1; }; \
	[ -n "$$P50_LINE" ] || { echo "❌ p50 line missing (classes)"; exit 1; }; \
	echo "✅ java_test (classes) passed"
