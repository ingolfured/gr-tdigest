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

CARGO   ?= cargo
POETRY  ?= poetry
JAVAC   ?= javac
JAVA    ?= java
PYTHON  ?= python3

JAVA_SRC := src-java
LIB_DIR  := target/release
CP_DEV   := target/java-classes

# Version (from Cargo.toml)
VER := $(shell sed -n 's/^version\s*=\s*"\(.*\)"/\1/p' Cargo.toml | head -1)

# Platform detection for JAR classifier + native filename
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

# Prebuilt natives for cross-packaging (optional)
#   NATIVE_LINUX_X86_64=/abs/path/libtdigest_rs.so
#   NATIVE_MACOS_AARCH64=/abs/path/libtdigest_rs.dylib
#   NATIVE_WINDOWS_X86_64=/abs/path/tdigest_rs.dll
NATIVE_LINUX_X86_64    ?=
NATIVE_MACOS_AARCH64   ?=
NATIVE_WINDOWS_X86_64  ?=

# CLI binary
CLI_BIN  ?= tdigest-rs
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
	@printf "  %-18s %s\n" "rust-cli-smoke" "Smoke: echo '0 1 2 3' | tdigest-rs quantile -q 0.5 == 1.5"
	@printf "\n$(STYLE_BOLD)Python$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "py-build"     "maturin develop -r -F python"
	@printf "  %-18s %s\n" "py-test"      "pytest -q"
	@printf "\n$(STYLE_BOLD)Java/JNI$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "java-build"   "Build JNI lib + compile Java classes to target/java-classes"
	@printf "  %-18s %s\n" "java-test"    "Run Java demo from classes (uses -Djava.library.path)"
	@printf "\n$(STYLE_BOLD)Publish$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "wheel"        "Build Python wheel (abi3)"
	@printf "  %-18s %s\n" "jar"          "Build API JAR + current-platform native JAR"
	@printf "  %-18s %s\n" "jar-all"      "Build API JAR + Linux/macOS/Windows native JARs (env points to prebuilt natives)"
	@printf "  %-18s %s\n" "verify-jar"   "Checksum + JNI-symbol check of embedded native"

# ==============================================================================
# Setup
# ==============================================================================
.PHONY: setup
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)
	$(call need,poetry)
	$(POETRY) --version
	$(call banner,Create Poetry venv + install deps)
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) env use 3.12 || true
	$(POETRY) install --with dev -q
	$(POETRY) run pip -q install -U maturin
	$(call banner,Quick Python import smoke)
	$(POETRY) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"

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
	$(POETRY) run ruff format .

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	$(POETRY) run ruff check .

clean:
	$(call banner,Clean: Rust target/)
	$(CARGO) clean
	$(call banner,Clean: Java classes & jars)
	rm -rf $(CP_DEV) $(JAVA_PKG_DIR) $(STAGE) || true
	$(call banner,Clean: Python artifacts)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -f tdigest_rs/*.so || true
	@printf "$(STYLE_OK)✓ cleaned Rust, Python, and Java artifacts$(STYLE_RESET)\n"

test: rust-test rust-cli-smoke py-test java-test
	@echo "✅ all tests passed"

help-me-run:
	@printf "\n$(STYLE_BOLD)Java — run from JARs (recommended for consumers)$(STYLE_RESET)\n"
	@printf "  make jar\n"
	@printf "  java -cp target/java/tdigest-rs-java-$(VER).jar:target/java/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar TestRun\n"
	@printf "  # Natives.load() extracts the embedded native from META-INF/native/$(PLAT)-$(ARCH)/ and loads it.\n"
	@printf "\n$(STYLE_BOLD)Java — run from classes (developer mode)$(STYLE_RESET)\n"
	@printf "  make java-build\n"
	@printf "  java -Djava.library.path=target/release -cp target/java-classes TestRun\n"
	@printf "\n$(STYLE_BOLD)Rust — CLI$(STYLE_RESET)\n"
	@printf "  cargo build --release --bin $(CLI_BIN)\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) quantile -q 0.5   # -> 1.5\n"
	@printf "  echo '0 1 2 3' | target/release/$(CLI_BIN) --probes '0,1.5,3' cdf\n"
	@printf "\n$(STYLE_BOLD)Python — module$(STYLE_RESET)\n"
	@printf "  poetry run python - <<'PY'\n"
	@printf "import tdigest_rs as pt\n"
	@printf "xs = [0.0, 1.0, 2.0, 3.0]\n"
	@printf "# TDigest is factory-constructed:\n"
	@printf "d = pt.TDigest.from_array(xs, max_size=100, scale='k2')\n"
	@printf "print('p50 =', d.quantile(0.5))\n"
	@printf "print('cdf =', d.cdf([0.0, 1.5, 3.0]).tolist())\n"
	@printf "PY\n"
	@printf "\n$(STYLE_BOLD)Polars — plugin$(STYLE_RESET)\n"
	@printf "  poetry run python - <<'PY'\n"
	@printf "import polars as pl, tdigest_rs as ptd\n"
	@printf "df = pl.DataFrame({'x':[0.0,1.0,2.0,3.0]})\n"
	@printf "df = df.select(ptd.tdigest('x', 100))\n"
	@printf "print(df.select(ptd.estimate_cdf('x', [0.0,1.5,3.0])))\n"
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
	OUT="$$( echo '0 1 2 3' | '$(CLI_PATH)' quantile -q $(Q) )"; \
	echo "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ rust_cli_smoke passed"

# ==============================================================================
# Python
# ==============================================================================
.PHONY: py-build py-test
py-build:
	$(POETRY) run maturin develop -r -F python

py-test: py-build
	$(POETRY) run pytest -q

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

# ==============================================================================
# Publish — Python wheel + Java JARs
# ==============================================================================
.PHONY: wheel jar jar-all verify-jar \
        api-jar native-jar-current \
        native-jar-linux-x86_64 native-jar-macos-aarch64 native-jar-windows-x86_64

wheel:
	$(POETRY) run maturin build --release --manylinux 2_28 --zig -F python

# Build API JAR + current platform native JAR
jar: api-jar native-jar-current
	@printf "$(STYLE_OK)✓ Built JARs under $(JAVA_PKG_DIR)$(STYLE_RESET)\n"

# API JAR (classes only) with a small manifest
api-jar: java-build
	$(call banner,Build API JAR)
	mkdir -p $(JAVA_PKG_DIR)
	printf "Manifest-Version: 1.0\n"                         >  $(API_MANIFEST)
	printf "Implementation-Title: tdigest-rs-java\n"         >> $(API_MANIFEST)
	printf "Implementation-Version: $(VER)\n"                >> $(API_MANIFEST)
	printf "Automatic-Module-Name: gr.tdigest_rs\n"         >> $(API_MANIFEST)
	jar --create --file $(API_JAR) \
	    -C $(JAVA_PKG_DIR) MANIFEST.MF \
	    -C $(CP_DEV) .

# Current platform native JAR (uses locally built native)
native-jar-current: java-build
	$(call banner,Build native JAR for $(PLAT)-$(ARCH))
	rm -rf $(STAGE)
	mkdir -p $(STAGE)/META-INF/native/$(PLAT)-$(ARCH)
	cp $(LIB_DIR)/$(LIBBASENAME) $(STAGE)/META-INF/native/$(PLAT)-$(ARCH)/
	jar --create --file $(NATIVE_JAR_CUR) \
	    -C $(STAGE) META-INF/native/$(PLAT)-$(ARCH)/$(LIBBASENAME)
	@printf "$(STYLE_OK)✓ %s$(STYLE_RESET)\n" "$(NATIVE_JAR_CUR)"

# Build all platform JARs (requires prebuilt natives via env vars)
jar-all: api-jar native-jar-current \
         native-jar-linux-x86_64 native-jar-macos-aarch64 native-jar-windows-x86_64
	@printf "$(STYLE_OK)✓ Built all JARs under $(JAVA_PKG_DIR)$(STYLE_RESET)\n"

native-jar-linux-x86_64:
	@if [ -z "$(NATIVE_LINUX_X86_64)" ]; then \
	  echo "$(STYLE_ERR)✗ Set NATIVE_LINUX_X86_64=/abs/path/libtdigest_rs.so$(STYLE_RESET)"; exit 1; \
	fi
	rm -rf $(STAGE)
	mkdir -p $(STAGE)/META-INF/native/linux-x86_64
	cp "$(NATIVE_LINUX_X86_64)" $(STAGE)/META-INF/native/linux-x86_64/libtdigest_rs.so
	jar --create --file $(JAVA_PKG_DIR)/tdigest-rs-java-$(VER)-linux-x86_64.jar \
	    -C $(STAGE) META-INF/native/linux-x86_64/libtdigest_rs.so

native-jar-macos-aarch64:
	@if [ -z "$(NATIVE_MACOS_AARCH64)" ]; then \
	  echo "$(STYLE_ERR)✗ Set NATIVE_MACOS_AARCH64=/abs/path/libtdigest_rs.dylib$(STYLE_RESET)"; exit 1; \
	fi
	rm -rf $(STAGE)
	mkdir -p $(STAGE)/META-INF/native/macos-aarch64
	cp "$(NATIVE_MACOS_AARCH64)" $(STAGE)/META-INF/native/macos-aarch64/libtdigest_rs.dylib
	jar --create --file $(JAVA_PKG_DIR)/tdigest-rs-java-$(VER)-macos-aarch64.jar \
	    -C $(STAGE) META-INF/native/macos-aarch64/libtdigest_rs.dylib

native-jar-windows-x86_64:
	@if [ -z "$(NATIVE_WINDOWS_X86_64)" ]; then \
	  echo "$(STYLE_ERR)✗ Set NATIVE_WINDOWS_X86_64=/abs/path/tdigest_rs.dll$(STYLE_RESET)"; exit 1; \
	fi
	rm -rf $(STAGE)
	mkdir -p $(STAGE)/META-INF/native/windows-x86_64
	cp "$(NATIVE_WINDOWS_X86_64)" $(STAGE)/META-INF/native/windows-x86_64/tdigest_rs.dll
	jar --create --file $(JAVA_PKG_DIR)/tdigest-rs-java-$(VER)-windows-x86_64.jar \
	    -C $(STAGE) META-INF/native/windows-x86_64/tdigest_rs.dll


verify-jar: jar
	$(call banner,Verify embedded native matches built native)
	@set -euo pipefail; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	orig_dir="$$(pwd)"; \
	jar_abs="$$(cd "$$orig_dir" && readlink -f "$(NATIVE_JAR_CUR)")"; \
	[ -f "$$jar_abs" ] || { echo "$(STYLE_ERR)✗ JAR not found: $$jar_abs$(STYLE_RESET)"; exit 1; }; \
	( cd "$$tmp_dir" && jar xf "$$jar_abs" META-INF/native/$(PLAT)-$(ARCH)/$(LIBBASENAME) ); \
	embedded="$$tmp_dir/META-INF/native/$(PLAT)-$(ARCH)/$(LIBBASENAME)"; \
	built="$(LIB_DIR)/$(LIBBASENAME)"; \
	echo "sha256 (embedded) vs (built):"; \
	sha256sum "$$embedded" "$$built"; \
	echo; echo "JNI symbols inside embedded native:"; \
	if [ "$(PLAT)" = "linux" ]; then \
	  nm -D "$$embedded" | grep -E 'Java_gr_tdigest_1rs_TDigestNative_' || { echo "❌ JNI exports missing"; exit 1; }; \
	elif [ "$(PLAT)" = "macos" ]; then \
	  nm -gU "$$embedded" | grep -E 'Java_gr_tdigest_1rs_TDigestNative_' || { echo "❌ JNI exports missing"; exit 1; }; \
	else \
	  nm "$$embedded" | grep -E 'Java_gr_tdigest_1rs_TDigestNative_' || { echo "❌ JNI exports missing"; exit 1; }; \
	fi; \
	echo "✅ verify-jar passed"
