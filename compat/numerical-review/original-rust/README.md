# Original Rust preservation evidence

The committed-lock oracle and the restored public package test different
things. This bundle keeps their provenance separate. No original Rust tests
exist at the pinned revision; the Rust API and numerical tests here are new
regressions, not an untouched upstream Rust suite.

## Original committed-lock oracle

Pinned upstream: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`.
Original Cargo.lock SHA-256:
`f37b4bc2afb2fc07c0e7c9f2787cd6cedd9b82fea724e3ad536ddaf679c4ac76`.
Fixture source: `52f6212008111b62fa07374278b850868b9ea77e`.
Existing 70-case fixture SHA-256:
`517ec82ec124e55920c7049a95f0eb0c9f3e96cf60b4ee7d4457bf31b7a98560`.

Both release and debug passed all 70 cases: 1,762 exact finite/infinity/signed-
zero bit comparisons plus 126 NaN classifications per profile, with no
differences. All 35 original files, including Cargo.toml and Cargo.lock,
remained byte-identical before and after. The only added source was a Rust
example with a dependency-free JSON serializer: no serde, no dependency or
manifest edits, and no fixture regeneration.

Portable preparation (Git checkouts must contain the named commits):

```sh
python oracle/prepare_oracle.py --upstream-repo /path/to/tdigest-rs \
  --modern-repo /path/to/gr-tdigest --workdir /tmp/tdigest-original-oracle
python /tmp/tdigest-original-oracle/run_oracle.py --run --profiles release debug
```

The preparer verifies every pinned file hash and the existing fixture hash
before writing a new disposable tree. The runner verifies them before and
after every `cargo run --locked`, using Cargo/Rust from PATH. Add `--offline`
only when the required original dependencies are cached. The original
historical runner used `/usr/sbin/cargo`; its SHA is recorded separately.
The portable runner uses PATH tool discovery and additionally verifies the two
original symlink targets, retaining the same comparison and serializer.
Packaging verified its syntax/preparation, not a new build.

`oracle/recorded/results.json` preserves exact original commands, compiler,
stdout and binary hashes. Binaries/outputs and source export are intentionally
omitted. The plan nested inside that completed result retains its historical
pre-run status; the enclosing `success: true` and profile results are the
execution outcome.

## Restored named Rust package

Tested implementation commit: `3b8fa69b9228be596d8733f58cdc39d780002b83`,
parent `7d9e13250772f7bbe7ae2433d3401efe326e07f3`.
Workspace lock SHA-256:
`046756d3a59e1f573807a5c613a7ebca158b634d1b96c360bbd1ee565790c7ae`.
The narrow tests ran on the uncommitted changes later captured exactly by that
commit. Thus `narrow-gates.json.base` records the parent, while
`committed-candidate.json` hashes the tested files in the final commit.

`tdigest_core` 0.2.0 is a workspace package at `src/upstream`, using the pinned
calculations with the three documented namespace-import adaptations through
the same private adapter alias. This is distinct from the byte-identical
original tree used by the committed-lock oracle above. It preserves public
`means: Vec<T>` and `weights: Vec<u32>` and all seven methods: from_array,
from_means_weights, quantile, median, trimmed_mean, merge, n_zero_weights.
Tests cover f32/f64 types, struct literals, mutation/destructuring, typed
anyhow errors, empty/error behavior and immutable merge semantics. The
separate numerical regression replays all 70 independently generated cases.

Both profiles passed the package API tests and numerical fixture tests.
`cargo check --features python --lib` passed. `cargo package -p tdigest_core`
passed; the unpacked artifact passed both profiles. Separate consumers of
the workspace path and unpacked package each passed both test profiles and
the release executable. Registry dependency versions/checksums matched the
workspace lock. An actual Maturin sdist contained the correct active nested
source and normalized path; the sdist SHA/layout is recorded, not bundled.
These are narrow package checks, not a substitute for later full wheel gates.

Reproduction from a clean disposable checkout of the tested commit:

```sh
cargo test --locked --offline -p tdigest_core
cargo test --locked --offline --release -p tdigest_core
cargo test --locked --offline --test original_rust_numerics
cargo test --locked --offline --release --test original_rust_numerics
cargo check --locked --offline --features python --lib
cargo package --locked --offline -p tdigest_core
```

`api/recorded/path-specific-runners/` are exact historical audit helpers,
included for review of artifact extraction and consumer checks; they contain
absolute machine paths and are not portable launchers. Use their explicit
commands/data flow in a disposable checkout after adapting the two top-level
paths. The original narrow package command included `--allow-dirty` because
the reviewed candidate was not yet committed. Consumer manifests/locks/main
and the tested API fixture are included. For a recreated consumer, seed the
unpacked package lock, resolve only that consumer node offline, verify every
registry version/checksum against the pinned workspace lock, then run tests
with `--locked`. Do not refresh all dependencies.

The original committed lock, independent fixture-generator lock and restored
workspace lock are intentionally different. `dependency-provenance/versions.json`
records the attribution; preserving source bytes alone is not a full lock
claim. The original-lock oracle independently confirms the same goldens under
the original graph.

Named Git/package API is preserved; a path consumer naming the root directory
must instead select `src/upstream`. Root package is still `gr-tdigest`.
No registry publication or ownership claim is part of this evidence; a future
registry release must publish the dependency package before the root crate.
