# TDigest handoff for the Ingo remote

Snapshot: 21 October 2026. Intended destination: the user's **Ingo** remote in
Codex, not Arch's box. This file is a portable project handoff. It does not
change repository permissions or transfer machine-local credentials/artifacts.

## Start here

Repository: https://github.com/ingolfured/gr-tdigest

Read `AGENTS.md`, `api_design.md`, `tdigest_design.md`, then
`tests/quality/ACCURACY.md`. Inspect the destination checkout's status and refs
before changing anything. Reuse a suitable checkout or clone into a new directory;
do not reset or overwrite unrelated work.

The complete current review source is:

- Branch: `review/20261021/37-accuracy-story`
- Commit: `556b2d2dd915b9d92a00143a05367da6484f65dd`
- Qualified runtime/artifact source:
  `aaf7ab3956fed4269c1eaeef915a6faee78fa305`
- Later changes after that runtime source are verification tools, documentation,
  lint wiring and receipts; core, Python binding, Java runtime and dependencies
  are unchanged. Do not claim binaries were rebuilt from the later documentation
  commit.

The review stack is 37 open, dependent personal PRs, #18 through #54. All their
GitHub checks passed at this snapshot. Recheck current heads/status before
continuing. Earlier #1/#2 were already merged; #3 through #17 are closed or
superseded. None of #18 through #54 was merged during this session.

Upstream G-Research/tdigest-rs PR #38 is a separate open draft at
`57e7978d0811baed4a546ff2f08dccaeff03c321`; it was not changed here.

## Immediate review-organization work

The user asked whether the PR split is good before reading it. The latest
assessment recommends three focused cleanups, **not yet implemented**:

1. Personal #25 combines compact original storage and optional Polars imports.
   Separate those independent concerns.
2. Personal #40 is the quality-suite PR but includes JVM bootstrap/helper fixes.
   Move those changes into the appropriate earlier JVM PRs (#37/#38).
3. Personal #48 combines original-oracle verification, measurement tools and
   build/package gates. Separate those review topics.

The rest of the split is broadly sensible. #27/#28 are large mostly because of
independent fixtures. #41 has many files because checked Rust construction must
update callers coherently. Tiny focused fixes need not be combined just to
increase their line counts. #53 is mostly recorded evidence.

Preserve original refs and source-bound evidence before reorganizing history.
Prove the final file tree is identical after a pure reorganization; run the
required build/test checks on each new review unit. Retain useful archive refs,
independent oracle and original-output fixtures. Update PR titles/descriptions
around each final scope, using clear prose and ASCII diagrams. Recompute the
PR additions/deletions table afterwards. Avoid adding a new general research
campaign when the remaining task is review organization.

**Merging is not confirmed.** A previous "do it" could refer to a smoke test or
merging all 37 PRs. A clarification was requested, and the user then asked for
PR tables and the split assessment. The public smoke test was completed; no
PR was merged. Do not interpret this handoff as permission to merge or publish.
The personal repository currently permits squash merges only; no settings were
changed. Preserve historical commit identities if choosing a later merge plan.

## Current PR map

Each PR's base is the preceding story branch; #18 starts from master.

| PR | Scope |
| --- | --- |
| 18 | Setup in worktrees and paths containing spaces |
| 19 | Weighted queries and restored state |
| 20 | Weighted tails, protected edges and wire settings |
| 21 | Float64 mass with either mean precision |
| 22 | Exact points and finite cluster bounds |
| 23 | Tiny-mass queries and failure-atomic scaling |
| 24 | Original-compatible Python implementation/API |
| 25 | Compact original storage and optional Polars imports |
| 26 | Unchanged upstream Python tests |
| 27 | Independent exact-output fixtures |
| 28 | Historical original pickle restoration |
| 29 | Clean Python source archives |
| 30 | Unchanged original Rust API |
| 31 | Installed-wheel runtime/Polars-free checks |
| 32 | K2Norm range handling without data loss |
| 33 | Checksum-bound source-byte protection |
| 34 | NumPy output-copy optimization |
| 35 | Polars delta configuration validation |
| 36 | NumPy training conversion and independent constructor concurrency |
| 37 | Verified JDK/Gradle bootstrap |
| 38 | Java 11, exclusive dependency sources and offline builds |
| 39 | Python 3.14/3.15 and modern NumPy/Polars stack |
| 40 | Recorded-data and deterministic quality suite |
| 41 | Checked modern core, transforms and full state |
| 42 | Modern Python operations and independent state copies |
| 43 | Polars cardinality and binary-schema rules |
| 44 | JVM operations, lifecycle, serialization and errors |
| 45 | Standalone native loading and identity |
| 46 | Authenticated Scala/Kotlin consumers |
| 47 | Installed Python typing/API documentation |
| 48 | Oracle, measurement tools and delivery gates |
| 49 | Public guides and unique development version |
| 50 | Earlier candidate artifact/evidence receipt |
| 51 | Count-only compression trials: performance fix |
| 52 | Real separate-process Spark consumer checks |
| 53 | Updated qualification evidence and intranet owner tools |
| 54 | Measured accuracy gains and explained tradeoffs |

Personal PR links: `https://github.com/ingolfured/gr-tdigest/pull/NUMBER`.

## Decisions already made by the user

- Original `tdigest_rs` Python and `tdigest_core` Rust must retain the pinned
  original API and arithmetic at
  `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Keep the original Python default
  **delta 300**. Do not substitute the modern engine or apply modern strictness.
- Modern defaults to **capacity 1,000**. Earlier modern capacity 200 is not the
  default/migration comparison target. Different delta/capacity values do not
  describe the same compression control.
- Modern callers may migrate to current APIs and dependencies, including
  approved Polars 2.0.0. There is no requested earlier-modern compatibility adapter.
- Modern missing digests and mixed capacities/configurations are rejected,
  including empty contributors. Existing valid empty state stays distinct.
- Binary Polars columns require explicit f32/f64 precision when decoding; this
  is not a conversion. Group-reduction cardinality stays documented.
- Practical accuracy means **no obvious, unexplained deterioration**. Publish
  separate position/rank, value, CDF, tail and storage scores. Explained
  approximation/merge/storage tradeoffs are acceptable; winning every metric
  on every profile is not required. Show measured strengths prominently.
- Hard numerical/state correctness, failure atomicity, exact original
  compatibility and frozen regression fixtures/gates stay strict and unchanged.
- Java 11/17/21 and Scala 2.12.16 matter. Spark 3.5 / Scala 2.12 is the provisional
  consumer family. Kotlin is a standalone JVM example, not the Spark language.
- Python 3.12/3.14/3.15 support is required; original compatibility also retains
  its tested 3.10/3.11 coverage. Exact tested NumPy cells are recorded; no blanket
  NumPy 2.x guarantee. The current modern Polars pin is 2.0.0.
- Python wheel floor: Linux x86-64 glibc 2.28. JNI floor: Linux x86-64 glibc 2.17,
  system allocator. Java bytecode major 55 alone does not prove the native floor.

## Evidence and measured strengths

Read these source-controlled locations:

- `compat/accuracy-20261021/` and `tests/quality/ACCURACY.md`: all 396 default
  profiles, 234 comparable-storage rows, 204 query reconstructions, plots,
  source/native/data hashes and reproducible scripts.
- `compat/followups-20261021/`: exact candidate runtime matrix, numerical controls,
  canonical timing comparison, source-bound metadata and ownership/offline checks.
- `compat/spark-20261021/`: four exact-JAR Spark consumer receipts.
- `review/20261021/`: earlier candidate receipt; historical evidence remains bound
  to its own artifact/source identity and is not silently relabelled.
- `src/upstream/PROVENANCE.md`, `scripts/check_upstream_sources.py` and
  `bindings/python/tests/data/`: original-source and independent-output guards.

Key results, without changing arithmetic or test gates:

- Default comparison: original delta 300 vs modern capacity 1000; same observations
  and controlled precision. Average position error is 98.7% lower across 396
  profiles, or 99.6% lower on 108 recorded-data profiles. Storage differs.
- Comparable logical storage: modern has lower average position error in
  213/234 eligible profiles and lower tail position error in 223, with 11 tail
  ties. Value-score losses remain visible and explained.
- All 4,140 diagnostic states match between the count-only optimization and its
  control. Canonical packaged diagnostic rows independently match exactly.
  The 114 small-capacity diagnostic misses and ten extended-suite misses remain
  recorded. Their labels are not a demand for universal metric dominance.
- All 15 exact numerical boundary controls pass. Numeric-range sweep: 204 states
  built, six expected overflow rejections, 1,692 successful transforms, 348 explicit
  rejected transforms and no unexpected failures.
- Fresh exact-wheel timings: streaming takes 16.5% less time and 40-way merging
  9.3% less than the preceding candidate. Against public modern source 1451f94,
  reductions are 5.7% and 4.4%. These are bounded modern-to-modern timings, not
  original-engine speed claims. No memory improvement is claimed.
- Same wheel: 12 installed runtime/dependency cells plus three source builds,
  totalling 6,285 repeated test executions. Actual glibc 2.28 runs pass.
- Same JAR: Java 11/17/21 and actual glibc 2.17 Java 11 runs pass. Scala 2.12.16 and
  Kotlin authenticated resolution, bad-credential rejection and staged-only
  offline runs pass. Exact byte/state cross-language, PyArrow24, typing and
  import ownership/upgrade/rollback checks pass.
- Spark 3.5.9, runtime Scala 2.12.18, compile Scala 2.12.16: Java 11/17 x Java/Kryo
  serializer checks pass using two separate executor JVMs, including broadcast,
  shuffle and checkpoint restoration. Kryo explicitly serializes TDIG state.
- A fresh public smoke test installs the exact prebuilt wheel outside the checkout,
  uses all 17,379 bike-data rows in 8 groups through lazy Polars/CDF and Arrow/Parquet,
  checks independent bytes/pickle/copies, and exercises Java 11 object-stream
  restoration after closing originals. It passed; this is not an intranet test.

## Exact prebuilt artifacts and access limits

Qualified candidate versions:
Python `0.2.5.dev2026102102`, Rust/JVM `0.2.5-dev2026102102`.

| Artifact | SHA-256 |
| --- | --- |
| `gr_tdigest-0.2.5.dev2026102102-cp310-abi3-manylinux_2_28_x86_64.whl` | `31bbc39f962dfbcd572cfbc296cf767cf4a01f7001591bb5936742b1dee162f9` |
| `gr-tdigest-0.2.5-dev2026102102.jar` | `6ec33b14d0add563fa6d11569d5585383d0d8a8a4f543bf7edb7cf59bba863cc` |
| Python native library | `1eaaf244db7004a0b8d2aca2edc6ff027a0d3a8871c2b8c450fdf94f19701d76` |
| JNI native library | `f254f1ff7b619e62fcb9879ad1a31593d74dcd01ea89361e0e3f1d5581484fc2` |

On the source machine only, the workspace is
`/home/ingo/Documents/ChatGPT/gr-tdigest 2`, with worktree `gr-tdigest-story` and
operational ledger `followups-20261021/STATE.json`.

Machine-local archives there include `followups-20261021/gr-tdigest-followups-20261021.zip`
(SHA256 `d4b8503ca9df8c0be50d786b2fe51c0965042acf35e81210756280c1d2c4c6e4`)
and `followups-20261021/accuracy-review-20261021.zip`. These paths are **not a
remote transfer**. The Git handoff gives access to source and committed evidence;
exact local wheel/JAR bytes require a separate transfer if needed. A rebuilt
artifact or a different CI wheel is not automatically the qualified same bytes.
Use a new unique candidate version for subsequent changed release candidates.

The private 0.1.9/0.1.11 consumer wheels, private fixtures and real internal
services will never be available externally. The user explicitly confirmed this.
Do not request their files, paths or credentials again. Those checks, actual
consumer executor images/configuration, internal publication and deployment
remain owner-run. Public tooling is in `tests/quality/OWNER_CHECKS.md`.

The historical SSD checkout at `/run/media/ingo/Storage/git/gr-tdigest` was
unavailable on the source machine. Preserve historical branches/experiments
until a source/evidence comparison is possible. Keep the handoff/archive refs,
including `archive/20261021/pre-story-candidate` at
`5801881075c6e31fec746c8c62c6a25782b6c13f` and the research handoff history.

## Working constraints

- No merges, production release/publication or deployment is authorized by this
  handoff alone. Do not change repository merge settings.
- Do not rewrite or relax exact original fixtures, source guards or frozen
  numerical gates to make a result pass. Preserve statistical mass and separate
  f32/f64 mean precision from Float64 mass.
- Follow `AGENTS.md`: read design documents, keep CHANGELOG updated per commit,
  and finish `make build`, `make test` and relevant lint/rustdoc checks.
- Coordinate heavy builds/timings and control nested pools. The source host has
  16 physical/32 logical CPUs and 59 GiB RAM, one NUMA node; visible CPU quota was
  unavailable. Do not infer the destination topology or claim 100-plus-CPU evidence.
- Do not restart the completed overnight automation or create background work
  beyond the user's current request. The old report/PDF/Drive work was completed.
- No private input access is expected from the external development machine.
  Preserve explicit passed, failed, historical and untested distinctions.
