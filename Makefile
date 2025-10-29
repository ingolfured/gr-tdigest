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
STYLE_CODE  := $(shell tput setaf 6 2>/dev/null || printf '\033[36m')

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null 2>&1 || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef
define sep
	@printf "\n$(STYLE_BOLD)====================$(STYLE_RESET)\n"
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

# Java sources
JAVA_SRC := bindings/java

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
CP_DEV   := target/java-classes   # compiled .class output

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

# Python distributions
DIST      ?= dist
DIST_DEV  ?= dist-dev

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
.PHONY: help setup clean \
        build build-rust build-python build-java \
        test rust-test py-test \
        lint \
        help-me-run \
        smoke-rust-cli smoke-wheel smoke-java \
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
	$(UV) --version
	$(call banner,Create .venv and install deps (all groups))
	$(UV) python install 3.12 || true
	$(UV) sync --all-groups
	$(call banner,Quick Python import smoke)
	$(UV) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"

# ==============================================================================
# Build (dev by default)
# ==============================================================================
build: build-rust build-python build-java
	@printf "$(STYLE_OK)✓ all components built (PROFILE=dev)$(STYLE_RESET)\n"

build-rust:
	$(call banner,Build: Rust lib + CLI ($(PROFILE)))
	$(CARGO) build $(CARGO_PROFILE_FLAG)
	$(CARGO) build $(CARGO_PROFILE_FLAG) --bin $(CLI_BIN)

build-python:
	$(call banner,Build: Python extension (maturin develop, dev))
	$(UV) run maturin develop -F python

build-java:
	$(call banner,Build: Java classes + native lib ($(PROFILE)))
	$(call need,javac)
	$(call need,java)
	$(CARGO) build $(CARGO_PROFILE_FLAG) --no-default-features --features java --lib
	rm -rf "$(CP_DEV)"; mkdir -p "$(CP_DEV)"
	javac -d "$(CP_DEV)" $$(find "$(JAVA_SRC)" -type f -name '*.java')

# ==============================================================================
# Tests
# ==============================================================================
test: rust-test py-test
	@echo "✅ all tests passed"

rust-test:
	$(CARGO) test -- --quiet

py-test: build-python
	$(UV) run pytest -q

# ==============================================================================
# Lint (autofix)
# ==============================================================================
lint:
	$(call banner,Format Rust)
	$(CARGO) fmt --all
	$(call banner,Rust: clippy --fix (may modify files))
	$(CARGO) clippy --fix --allow-dirty --allow-staged
	$(call banner,Python: ruff check --fix + ruff format)
	$(UV) run ruff check --fix .
	$(UV) run ruff format .


# ==============================================================================
# Clean (single target)
# ==============================================================================
clean:
	$(call banner,Clean: Rust target/ and cargo)
	rm -rf target/ || true
	$(CARGO) clean || true
	$(call banner,Clean: Java classes & jars)
	rm -rf "$(CP_DEV)" "$(JAVA_PKG_DIR)" "$(STAGE)" || true
	find "$(JAVA_SRC)" -type f \( -name '*.class' -o -name '*.java.bak' \) -delete || true
	$(call banner,Clean: Python build artifacts)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -f tdigest_rs/*.so tdigest_rs/*.dylib tdigest_rs/*.pyd tdigest_rs/*.dll || true
	rm -rf build/ *.egg-info/ || true
	$(call banner,Clean: Wheel dists)
	rm -rf "$(DIST)" "$(DIST_DEV)" || true
	@printf "$(STYLE_OK)✓ cleaned all artifacts$(STYLE_RESET)\n"

# ==============================================================================
# Help me run — one-liners (Demos)
# ==============================================================================
help-me-run:
	$(call sep)
	@printf '1) Rust — CLI (release)\n'
	@printf '%s\n' 'make release-rust'
	@printf '%s\n' "echo '0 1 2 3' | target/release/$(CLI_BIN) --stdin --cmd quantile --p 0.5 --no-header"

	$(call sep)
	@printf '2) Python — basic import from release wheel\n'
	@printf '%s\n' 'make release-wheel'
	@printf '%s\n' 'TMP=$$(mktemp -d)'
	@printf '%s\n' 'UV_LINK_MODE=copy uv venv --python 3.12 --seed $$TMP'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python --upgrade pip'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python "polars>=1.34.0,<2.0.0" numpy'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python --no-index --find-links dist tdigest_rs'
	@printf '%s\n' '$$TMP/bin/python -c '\''import tdigest_rs as td; d=td.TDigest.from_array([0,1,2,3], max_size=100, scale="k2"); print("p50=", d.quantile(0.5)); print("cdf=", d.cdf([0.0,1.5,3.0]).tolist())'\'''
	@printf '%s\n' 'rm -rf $$TMP'

	$(call sep)
	@printf '3) Polars — groupby + quantile via release wheel\n'
	@printf '%s\n' 'make release-wheel'
	@printf '%s\n' 'TMP=$$(mktemp -d)'
	@printf '%s\n' 'UV_LINK_MODE=copy uv venv --python 3.12 --seed $$TMP'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python --upgrade pip'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python "polars>=1.34.0,<2.0.0" numpy'
	@printf '%s\n' 'UV_LINK_MODE=copy uv pip install --python $$TMP/bin/python --no-index --find-links dist tdigest_rs'
	@printf '%s\n' '$$TMP/bin/python -c '\''import polars as pl; from tdigest_rs import tdigest, quantile; df=pl.DataFrame({"g":["a"]*5,"x":[0,1,2,3,4]}); out=(df.lazy().group_by("g").agg(tdigest(pl.col("x"), max_size=100, scale="k2").alias("td")).select(quantile("td",0.5)).collect()); print(out)'\'''
	@printf '%s\n' 'rm -rf $$TMP'
		$(call sep)
	@printf '4) Java — compile & run against release JAR + native lib\n'
	@printf '%s\n' 'make release-jar'
	@printf '%s\n' 'mkdir -p target/java-hello'
	@printf "%s\n" "printf '%s\n' \
'import gr.tdigest_rs.TDigest;' \
'import gr.tdigest_rs.TDigest.Precision;' \
'import gr.tdigest_rs.TDigest.Scale;' \
'import gr.tdigest_rs.TDigest.SingletonPolicy;' \
'import java.util.Arrays;' \
'public class HelloTDigest {' \
'  public static void main(String[] args) {' \
'    try (TDigest d = TDigest.builder().maxSize(100).scale(Scale.K2).singletonPolicy(SingletonPolicy.EDGES).keep(4).precision(Precision.F32).build(new float[]{0,1,2,3})) {' \
'      System.out.println(Arrays.toString(d.cdf(new double[]{0.0,1.5,3.0})));' \
'      System.out.println(\"p50 = \" + d.quantile(0.5));' \
'    }' \
'  }' \
'}' > target/java-hello/HelloTDigest.java"
	@printf '%s\n' 'javac -cp $(API_JAR) -d target/java-hello target/java-hello/HelloTDigest.java'
	@printf '%s\n' 'java --enable-native-access=ALL-UNNAMED -Djava.library.path=target/release -cp $(API_JAR):target/java-hello HelloTDigest'
	$(call sep)

# ==============================================================================
# Smoke tests (generic and independent)
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
	UV_LINK_MODE=copy $(UV) venv --python 3.12 --seed "$$TMP"; \
	UV_LINK_MODE=copy $(UV) pip install --python "$$TMP/bin/python" --upgrade pip; \
	UV_LINK_MODE=copy $(UV) pip install --python "$$TMP/bin/python" 'polars>=1.34.0,<2.0.0' numpy; \
	UV_LINK_MODE=copy $(UV) pip install --python "$$TMP/bin/python" --no-index --find-links "$$(dirname "$$LATEST")" tdigest_rs; \
	"$$TMP/bin/python" -c "import tdigest_rs as td; d=td.TDigest.from_array([0.0,1.0,2.0,3.0], max_size=100, scale='k2'); print('p50=', d.quantile(0.5)); print('cdf=', d.cdf([0.0,1.5,3.0]).tolist())"; \
	rm -rf "$$TMP"; \
	echo "✅ smoke-wheel ok (file=$$LATEST)"


smoke-java:
	@set -eu; \
	[ -f "$(API_JAR)" ] || { echo "❌ missing API JAR $(API_JAR); run 'make release-jar' (release) or 'make build-java && make jar' (dev)"; exit 1; }; \
	OUT="$$( $(JAVA) --enable-native-access=ALL-UNNAMED -Djava.library.path=$(LIB_DIR) -cp "$(API_JAR)" TestRun || true)"; \
	echo "• Java smoke output:"; echo "$$OUT"; \
	ARR_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^\[[[:space:]]*0\.125,\s*0\.5,\s*0\.875\]$$' || true)"; \
	P50_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^p50[[:space:]]*=[[:space:]]*1\.5$$' || true)"; \
	[ -n "$$ARR_LINE" ] || { echo "❌ array mismatch"; exit 1; }; \
	[ -n "$$P50_LINE" ] || { echo "❌ p50 line missing"; exit 1; }; \
	echo "✅ smoke-java ok (PROFILE=$(PROFILE))"

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
	$(UV) run maturin build -r -F python --compatibility manylinux_2_28 --zig -o "$(DIST)"
	@printf "$(STYLE_OK)✓ wheel(s) in $(DIST)$(STYLE_RESET)\n"
	$(MAKE) smoke-wheel

release-jar:
	$(call banner,Release: Java (release) + smoke)
	$(MAKE) build-java PROFILE=release
	$(call banner,Package Java API/NATIVE JARs (release libs))
	mkdir -p "$(JAVA_PKG_DIR)" "$(STAGE)"
	printf "Manifest-Version: 1.0\nAutomatic-Module-Name: gr.tdigest_rs\n" > "$(API_MANIFEST)"
	$(JAR) cfm "$(API_JAR)" "$(API_MANIFEST)" -C "$(CP_DEV)" .
	rm -rf "$(STAGE)/natives" && mkdir -p "$(STAGE)/natives/$(PLAT)-$(ARCH)"
	cp "target/release/$(LIBBASENAME)" "$(STAGE)/natives/$(PLAT)-$(ARCH)/$(LIBBASENAME)"
	$(JAR) cf "$(NATIVE_JAR_CUR)" -C "$(STAGE)" natives
	$(MAKE) smoke-java PROFILE=release
	@printf "• API JAR    -> %s\n" "$(API_JAR)"; \
	printf "• Native JAR -> %s\n" "$(NATIVE_JAR_CUR)"

release: release-rust release-wheel release-jar
	@printf "\n$(STYLE_BOLD)==> Release complete$(STYLE_RESET)\n"
	@printf "  CLI binary : %s\n" "target/release/$(CLI_BIN)"
	@printf "  Wheels dir : %s\n" "$(DIST)"
	@printf "  Native JAR : %s\n" "$(NATIVE_JAR_CUR)"
	@printf "$(STYLE_OK)✓ All release artifacts built & smoke-tested$(STYLE_RESET)\n"

help:
	@printf "\n$(STYLE_BOLD)Core (dev by default)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "setup"          "Install toolchains and sync Python deps (uv)"
	@printf "  %-22s %s\n" "build"          "Build Rust lib+CLI (dev), Python ext (dev), Java classes (dev)"
	@printf "  %-22s %s\n" "build-rust"     "Build Rust lib+CLI (dev)"
	@printf "  %-22s %s\n" "build-python"   "Build Python extension (maturin develop, dev)"
	@printf "  %-22s %s\n" "build-java"     "Build Java classes + native lib (dev)"
	@printf "  %-22s %s\n" "test"           "Run tests: rust + python"
	@printf "  %-22s %s\n" "lint"           "Autofix: cargo fmt, cargo clippy --fix, ruff --fix, ruff format"
	@printf "  %-22s %s\n" "clean"          "Remove ALL build artifacts, wheels, jars, caches"
	@printf "\n$(STYLE_BOLD)Releases (release profile)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "release-rust"   "Build CLI (release) and smoke"
	@printf "  %-22s %s\n" "release-wheel"  "Build PROD wheel (release) inline and smoke"
	@printf "  %-22s %s\n" "release-jar"    "Build JARs (release) and smoke"
	@printf "  %-22s %s\n" "release"        "release-rust + release-wheel + release-jar"
	@printf "\n$(STYLE_BOLD)Demos$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "help-me-run"    "One-liner demos using release artifacts"
