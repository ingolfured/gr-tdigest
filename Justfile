# Justfile — pretty front-end for Make targets
set shell := ["bash", "-uc"]

default:
    @just --list

# ---------------------------------------------------------------------------
# Environment setup / teardown
# ---------------------------------------------------------------------------
setup:
    make setup

resetup:
    make resetup

nuke:
    make nuke

# ---------------------------------------------------------------------------
# Tests / CI
# ---------------------------------------------------------------------------
test:
    make test

fulltest:
    make fulltest

bench:
    make bench_quick

smoke:
    make maturin_smoke

# ---------------------------------------------------------------------------
# Meta / release
# ---------------------------------------------------------------------------
ci:
    make ci_local

ship:
    make ship

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
fmt:
    make fmt

lint:
    make lint

check:
    make check

help:
    make help
