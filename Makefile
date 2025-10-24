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

CARGO  ?= cargo
UV     ?= uv
JAVAC  ?= javac
JAVA   ?= java
JAR    ?= jar
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

# CLI binary
CLI_BIN  ?= tdigest
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)

# Distributions
DIST ?= dist

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
.PHONY: help setup build fmt lint clean test help-me-run \
        rust-build rust-test rust-cli-smoke \
        py-build py-test wheel \
        java-build java-test jar release

# ==============================================================================
# HELP
# ==============================================================================
help:
	@printf "\n$(STYLE_BOLD)Core$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "setup"        "Install toolchains and Python env (uv), then smoke-import Polars"
	@printf "  %-18s %s\n" "build"        "Build Rust lib+CLI (with smoke), Python extension, and Java classes"
	@printf "  %-18s %s\n" "test"         "Run: Rust tests and Python tests"
	@printf "  %-18s %s\n" "fmt"          "Format Rust (rustfmt) and Python (ruff format)"
	@printf "  %-18s %s\n" "lint"         "Lint Rust (clippy -D warnings) and Python (ruff)"
	@printf "  %-18s %s\n" "clean"        "Remove Rust, Python, Java, and distribution artifacts"
	@printf "  %-18s %s\n" "help-me-run"  "Quick examples: Rust CLI, Python API, Polars exprs, Java demo"
	@printf "  %-18s %s\n" "release"      "Build & validate CLI (smoke), wheel (smoke), and JARs (smoke)"
	@printf "\n$(STYLE_BOLD)Rust$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "rust-build"   "cargo build --release (lib) + --bin $(CLI_BIN) (CLI), then smoke-test CLI"
	@printf "  %-18s %s\n" "rust-test"    "cargo test -- --quiet"
	@printf "\n$(STYLE_BOLD)Python$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "py-build"     "Editable install via maturin develop -r -F python"
	@printf "  %-18s %s\n" "py-test"      "pytest -q"
	@printf "  %-18s %s\n" "wheel"        "Build wheel(s) (manylinux_2_28 + zig) into $(DIST)/ and smoke-test install"
	@printf "\n$(STYLE_BOLD)Java/JNI$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "java-build"   "Build native lib (features=java) and compile Java classes to $(CP_DEV)/"
	@printf "  %-18s %s\n" "jar"          "Package API JAR + native JAR; runs Java smoke after packaging"

# ==============================================================================
# Setup
# ==============================================================================
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
build:
	$(call banner,Build: Rust lib + CLI (with smoke))
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
	rm -rf target/
	$(CARGO) clean || true

	$(call banner,Clean: Java classes & jars)
	rm -rf "$(CP_DEV)" "$(JAVA_PKG_DIR)" "$(STAGE)"
	find "$(JAVA_SRC)" -type f \( -name '*.class' -o -name '*.java.bak' \) -delete || true

	$(call banner,Clean: Python artifacts)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	# Native extension outputs (Linux/macOS/Windows)
	rm -f tdigest_rs/*.so tdigest_rs/*.dylib tdigest_rs/*.pyd tdigest_rs/*.dll || true
	rm -rf .venv-wheeltest || true

	$(call banner,Clean: Distribution packages)
	rm -rf "$(DIST)" build/ *.egg-info/ || true

	@printf "$(STYLE_OK)✓ cleaned Rust, Python, Java, and distribution artifacts$(STYLE_RESET)\n"

# Only Rust + Python tests in aggregate
test: rust-test py-test
	@echo "✅ all tests passed"

help-me-run:
	@printf "\n$(STYLE_BOLD)Rust — CLI$(STYLE_RESET)\n"
	@printf "  cargo build --release --bin $(CLI_BIN)\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) --stdin --cmd quantile --p 0.5 --no-header   # -> 1.5\n"
	@printf "  echo '0 1.5 3' | target/release/$(CLI_BIN) --stdin --cmd cdf                          # -> CSV x,p rows\n"
	@printf "\n$(STYLE_BOLD)Python — API$(STYLE_RESET)\n"
	@printf "  uv run python - <<'PY'\n"
	@printf "import polars as pl\n"
	@printf "import tdigest_rs as td\n"
	@printf "values = [0.0, 1.0, 2.0, 3.0]\n"
	@printf "d = td.TDigest.from_array(values, max_size=100, scale='k2')\n"
	@printf "print('p50 =', d.quantile(0.5))\n"
	@printf "print('cdf =', d.cdf([0.0, 1.5, 3.0]).tolist())\n"
	@printf "# Polars lazy example (requires exprs exported by extension)\n"
	@printf "try:\n"
	@printf "    from tdigest_rs import tdigest, quantile  # expr functions\n"
	@printf "    df = pl.DataFrame({'g':['a']*5, 'x':[0,1,2,3,4]})\n"
	@printf "    out = (\n"
	@printf "        df.lazy().group_by('g')\n"
	@printf "        .agg(tdigest(pl.col('x'), max_size=100, scale='k2').alias('td'))\n"
	@printf "        .select(quantile('td', 0.5))\n"
	@printf "        .collect()\n"
	@printf "    )\n"
	@printf "    print(out)\n"
	@printf "except Exception as e:\n"
	@printf "    print('Polars expr demo skipped:', e)\n"
	@printf "PY\n"
	@printf "\n$(STYLE_BOLD)Java — demo$(STYLE_RESET)\n"
	@printf "  make java-build && java -Djava.library.path=$(LIB_DIR) -cp $(CP_DEV) TestRun\n"

# ==============================================================================
# Rust
# ==============================================================================
rust-build:
	# Build full release (lib) and ensure CLI bin is built, then run CLI smoke
	$(CARGO) build --release
	$(CARGO) build --release --bin $(CLI_BIN)
	$(MAKE) rust-cli-smoke

rust-test:
	$(CARGO) test -- --quiet

Q ?= 0.5
$(CLI_PATH):
	$(CARGO) build --release --bin $(CLI_BIN)

# Hidden from help, but always used by rust-build/release
rust-cli-smoke: $(CLI_PATH)
	@set -eu; \
	OUT="$$( echo '0 1 2 3' \
	  | '$(CLI_PATH)' --stdin --cmd quantile --p $(Q) --no-header --output csv \
	  | cut -d, -f2 )"; \
	printf "CLI p%.3g -> %s\n" "$(Q)" "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ rust_cli_smoke passed"

# ==============================================================================
# Python
# ==============================================================================
py-build:
	$(UV) run maturin develop -r -F python

py-test: py-build
	$(UV) run pytest -q

# Wheels: build + smoke in one target
wheel:
	$(call banner,Build Python wheel (manylinux_2_28 + zig))
	mkdir -p "$(DIST)"
	$(UV) run maturin build -r -F python --compatibility manylinux_2_28 --zig -o "$(DIST)"
	@printf "$(STYLE_OK)✓ wheel(s) in $(DIST)$(STYLE_RESET)\n"
	$(call banner,Smoke test the wheel in a clean venv)
	rm -rf .venv-wheeltest
	$(UV) venv --python 3.12 --seed .venv-wheeltest
	.venv-wheeltest/bin/python -m pip --version || .venv-wheeltest/bin/python -m ensurepip --upgrade
	.venv-wheeltest/bin/python -m pip install --upgrade pip
	.venv-wheeltest/bin/python -m pip install "polars>=1.34.0,<2.0.0" numpy
	.venv-wheeltest/bin/python -m pip install --no-index --find-links "$(DIST)" tdigest_rs
	.venv-wheeltest/bin/python -c "import tdigest_rs as td; d=td.TDigest.from_array([0.0,1.0,2.0,3.0],max_size=100,scale='k2'); print('p50=',d.quantile(0.5)); print('cdf=',d.cdf([0.0,1.5,3.0]).tolist())"
	rm -rf .venv-wheeltest
	@echo "$(STYLE_OK)✓ wheel build + smoke ok$(STYLE_RESET)"

# ==============================================================================
# Java / JNI (classes build + jars)
# ==============================================================================
java-build:
	$(call need,javac)
	$(call need,java)
	$(CARGO) build --release --no-default-features --features java --lib
	rm -rf "$(CP_DEV)"
	mkdir -p "$(CP_DEV)"
	javac -d "$(CP_DEV)" $(shell find "$(JAVA_SRC)" -type f -name '*.java')

# Hidden from help; called by jar
java-test:
	@set -e; \
	OUT="$$( $(JAVA) -Djava.library.path=$(LIB_DIR) -cp $(CP_DEV) TestRun )"; \
	echo "• Java smoke output:"; \
	echo "$$OUT"; \
	ARR_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^\[[[:space:]0-9\.,]+\]$$' || true)"; \
	P50_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^p50[[:space:]]*=[[:space:]]*1\.5$$' || true)"; \
	[ "$$ARR_LINE" = "[0.125, 0.5, 0.875]" ] || { echo "❌ array mismatch (classes)"; exit 1; }; \
	[ -n "$$P50_LINE" ] || { echo "❌ p50 line missing (classes)"; exit 1; }; \
	echo "✅ java_test (classes) passed"

# Package API JAR (classes only) + native JAR (contains platform lib under /natives/<plat-arch>/)
jar: java-build
	$(call banner,Package Java API JAR)
	mkdir -p "$(JAVA_PKG_DIR)" "$(STAGE)"
	printf "Manifest-Version: 1.0\nAutomatic-Module-Name: gr.tdigest_rs\n" > "$(API_MANIFEST)"
	$(JAR) cfm "$(API_JAR)" "$(API_MANIFEST)" -C "$(CP_DEV)" .
	@printf "$(STYLE_OK)✓ API JAR -> %s$(STYLE_RESET)\n" "$(API_JAR)"

	$(call banner,Package platform native JAR)
	rm -rf "$(STAGE)/natives" && mkdir -p "$(STAGE)/natives/$(PLAT)-$(ARCH)"
	cp "$(LIB_DIR)/$(LIBBASENAME)" "$(STAGE)/natives/$(PLAT)-$(ARCH)/$(LIBBASENAME)"
	$(JAR) cf "$(NATIVE_JAR_CUR)" -C "$(STAGE)" natives
	@printf "$(STYLE_OK)✓ Native JAR -> %s$(STYLE_RESET)\n" "$(NATIVE_JAR_CUR)"

	$(call banner,Java smoke (post-jar))
	$(MAKE) java-test

# ==============================================================================
# One-shot RELEASE: build & smoke-test all artifacts
# ==============================================================================
release:
	$(call banner,Release: Build Rust CLI (with smoke))
	$(MAKE) rust-build
	@printf "• CLI -> %s\n" "$(CLI_PATH)"

	$(call banner,Release: Build Wheel (with smoke))
	$(MAKE) wheel
	# Capture latest wheel for summary (most recent by mtime)
	LAST_WHEEL="$$(ls -1t "$(DIST)"/*.whl 2>/dev/null | head -1 || true)"; \
	if [ -z "$$LAST_WHEEL" ]; then \
		printf "$(STYLE_ERR)✗ No wheel found in %s$(STYLE_RESET)\n" "$(DIST)"; exit 1; \
	fi; \
	printf "• Wheel -> %s\n" "$$LAST_WHEEL"; \
	echo "$$LAST_WHEEL" > .last-wheel-path

	$(call banner,Release: Build JARs (with smoke))
	$(MAKE) jar
	@printf "• API JAR    -> %s\n" "$(API_JAR)"
	@printf "• Native JAR -> %s\n" "$(NATIVE_JAR_CUR)"

	# Final clear artifact summary
	@LAST_WHL="$$(cat .last-wheel-path 2>/dev/null || true)"; \
	rm -f .last-wheel-path; \
	printf "\n$(STYLE_BOLD)==> Artifacts$(STYLE_RESET)\n"; \
	printf "  CLI binary : %s\n" "$(CLI_PATH)"; \
	printf "  Wheel      : %s\n" "$${LAST_WHL:-<none>}"; \
	printf "  Native JAR : %s\n" "$(NATIVE_JAR_CUR)"; \
	printf "\n$(STYLE_OK)✓ Release artifacts built & smoke-tested$(STYLE_RESET)\n"
