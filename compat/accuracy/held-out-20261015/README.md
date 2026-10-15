# Reproducing the held-out fixed-delta accuracy survey

The source is pinned to modern commit
`52f6212008111b62fa07374278b850868b9ea77e` and independent original commit
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. All work is outside tracked repositories.
The corrected tiny smoke and complete fixed-delta held-out survey executed on
15 October 2026 (local time); see `RESULTS.md`. The first smoke build failure is
also retained. No library source changed during this study. The frozen Rust
harness's initial 'not yet compiled/run' comment records its preparation state;
the hash-pinned file was deliberately not rewritten after execution.

Keep `run_held_out.py`, `held_out_accuracy.rs`, `accuracy_reference.py` and
`held_out_plan.py` together. `run_held_out.py` prints a plan unless `--execute` is
present. It records a frozen protocol containing all four source hashes, both
commit IDs, distributions/seeds/sizes, delta settings and route choices. Changing
one of those files invalidates earlier smoke evidence deliberately.

## Gates and bounded execution

The runner requires Linux/WSL, Python 3.12+, Rust 1.90.0, cached Cargo dependencies,
and an assigned CPU/cache slot. It exports the exact Git commit into a fresh output
directory, verifies the independently extracted original source hashes, and builds
a standalone release executable offline. It never checks out, resets or edits the
input repositories. A nonempty output directory is rejected.

Reserve the shared CPU/cache before building, then start with the small gate:

```sh
python3 run_held_out.py --execute --phase smoke \
  --repo /path/to/repository-containing-52f6212 \
  --upstream /path/to/independent/tdigest-rs \
  --output /path/to/fresh/accuracy-smoke \
  --cargo-target /path/to/reserved/release-cache
```

Before building, fourteen manually computed rational checks validate the Python
oracle. After building, `--reference-probe` exercises the actual Rust reference
helpers on seven fixed fixtures: singleton, two points, ties, negative/gapped
values, a small adjacent-scale point, an F32 rounding collision and an equal pile.
The independent Python oracle checks 763 returned numeric cells. This is followed
by 16 n=32 datasets: eight distributions in F64/F32, seed 1, producing 176 states.
Each tiny state includes its exact post-cast sorted input; every emitted Type-7,
tie-aware rank and midpoint-CDF reference is checked independently. Input query
rounding is also checked for both precisions.

The reference oracle uses `Fraction` for exact rational interpolation and linear
counts, independently of the Rust floating interpolation and binary partition
search. Probability comparisons allow four ULPs at unit scale for floating
arithmetic. Tiny Type-7 comparisons additionally bound position rounding and local
cancellation using adjacent gaps; they do not silently substitute the Rust result
as the expected value. The manually computed dyadic gold cases remain tight.

Every modern invariant failure is retained and makes the smoke fail. Inherited
original behavior remains in `inherited_original_observations` and the raw error
rows. It does not censor data or make the modern implementation fail. For example,
the old `[1,2]`, q=0.1 estimate of 0.7 remains inherited behavior to report.

Inspect `metadata.json` before progressing: status `complete`, phase `smoke`,
16 cases/176 states, manual/probe/tiny reference checks passed, no failures and no
source changes. All per-case files must match the recorded compressed-file hashes.
The executable is retained with its hash. Smoke does not start a full run.

Training or held-out execution requires the complete successful smoke directory
from the exact same protocol. Held-out execution also requires the previously
written protocol lock; it is not created by reading held-out results:

```sh
python3 run_held_out.py --execute --phase held-out \
  --repo /path/to/repository-containing-52f6212 \
  --upstream /path/to/independent/tdigest-rs \
  --output /path/to/fresh/accuracy-held-out \
  --cargo-target /path/to/reserved/release-cache \
  --smoke-evidence /path/to/fresh/accuracy-smoke \
  --protocol-lock /path/to/fresh/accuracy-smoke/protocol.json
```

The fixed-delta survey does not select parameters from training results: both
cores use delta 300, with modern K2Norm/singleton Off, as predeclared. Training is
available as a separate 240-dataset phase. A matched-memory selection experiment
requires a separate protocol/implementation; this runner must not be labelled a
matched-memory study merely because it records memory.

## Output, budgets and failure handling

Cases stream one JSON state at a time into individual `cases/*.jsonl.gz` files.
The runner does not retain the full output of a case or survey in memory. Stderr
has separate per-case logs. Metadata stores compact case summaries, raw-file
hashes, coverage, observations and failures, and is saved after each case.

The same-delta held-out/training phase has 240 datasets and 3,360 state records
(including reversed shard order for n=100,000). No operation timings are collected;
wall durations are only execution-budget metadata. Full raw output could approach
1 GB uncompressed, which is why it is compressed as written.

Default dataset-runtime budgets are 180 seconds for smoke and 900 seconds for a
survey. The budget is checked between cases; each case additionally has a 120-second
timeout, so the runtime bound is the budget plus at most one case timeout. Builds
have a separate 1,800-second timeout. Flags can adjust these explicitly after the
CPU owner reviews progress. These are caps, not measured duration promises.

A timeout preserves partial compressed data and records failure. A modern
invariant error or schema/reference error is recorded while later declared cases
continue within the budget; it is not dropped from the report. A budget-limited
run is marked `incomplete` and exits nonzero. A failed run cannot serve as smoke
evidence. Sources are rehashed at completion. Build/probe failures leave an
incomplete metadata status and logs, never a success claim.

The manual oracle, Rust reference probe, tiny smoke, full held-out run and
independent saved-file hash verification have passed for the recorded run.
Training was not run and no parameters were selected from held-out data. The
execution-time budgets above are implemented. Matched-memory parameter selection,
trimmed-mean metrics and comparison/error/storage-budget enforcement are not.
There is no universal accuracy threshold asserted by a passing invariant gate.

## Compact evidence and full raw data

`results/metadata.json` records all 240 compressed-case hashes and immutable
source/toolchain provenance. The complete raw case files, source exports and
executable remain in the original local run directory, identified by
`FULL-RAW-LOCATION.json`. They are not included or claimed to be committed in this
compact package. Preserve that directory for exact artifact inspection; rerun
from the pinned commits to regenerate a full dataset elsewhere. The two historical
Git commit objects must exist in the provided repositories; do not substitute HEAD.

The compact package includes every per-state metric and per-route original drift
as gzip JSONL, per-seed grouped summaries, exact notable cases, and eight unchanged
raw direct-state records. `selected-raw-records-sources.json` maps each extracted
line to its original case filename and hashes. Full raw records were not censored;
the smaller review extract is explicitly selected for inspection.

Smoke subdirectories here contain metadata and logs only. They are not complete
`--smoke-evidence` inputs: run a fresh complete smoke before reproducing the survey.
The original failed compile is retained alongside the corrected successful smoke.

`analyze_held_out.py /path/to/full/run` verifies the source/binary/Cargo.lock/protocol
and compressed-file hashes, then streams the gzip raw files to produce
`state-metrics.jsonl`, `paired-drift.jsonl` and `summary.json`.
`describe_held_out.py /path/to/full/run` prints the descriptive tables and saves
`notable-cases.json`. Both require a complete run directory, not this compact
package. Neither runs or times the library. Analyzers are post-run tools and are
separate from the frozen execution protocol.

Metric summaries are per state and equally weight the declared routes. The
quantile metric uses empirical tie intervals; the value metric uses Type-7;
CDF uses midpoint ECDF. State populations for modern and original differ because
only modern has the k-way route. Restrict to direct/sequential matching routes for
like-for-like comparison. Logical live payload is not allocated capacity or RSS.

The previous broader design is exploratory context only. This package executed
the fixed-delta, unweighted study stated in its protocol, not every planned future
experiment. It supports no global speed/accuracy ranking.
