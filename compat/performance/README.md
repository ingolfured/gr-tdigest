# Linux performance and accuracy evidence

The compatibility interface (`tdigest_rs`) keeps the original arithmetic. The modern engine (`gr_tdigest`) is opt-in. The useful story is measured tradeoffs: compatibility, much smaller reserved result capacity, and workload-dependent modern speed/accuracy improvements. These measurements do **not** establish a global speed or accuracy lead.

## Current committed evidence

The latest full run targets clean commit `22867bd55739a96a6fab3c8c48b27406ecb32197` and includes the f64-weight and single-centroid mean fixes. No source changed during compilation or timing. See [current results and limitations](RESULTS-22867bd.md), [raw samples](results/2026-10-14-22867bd.jsonl), [provenance](results/2026-10-14-22867bd-metadata.json) and [comparison with 756d221](results/2026-10-14-22867bd-comparison.json). The scalar-CDF optimization remains scratch-only.

Release Rust cores on Linux/WSL, Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0, delta=300, both precisions, seven batches per operation:

| Selected workload | Original → modern |
|---|---|
| Build 10M random uniform f64 values | 2.840 s → 0.687 s (~4.1× faster) |
| Build 10M random uniform f32 values | 2.186 s → 0.436 s (~5.0× faster) |
| Build 1K random uniform f64 values | 15.7 µs → 17.9 µs |
| Build 100K sorted f32 values | 930 µs → 1257 µs |
| Merge 40 uniform f64 shards, 100K total values | Original sequential 329.6 µs; modern sequential 564.7 µs; modern k-way 211.9 µs |
| Maximum sampled rank error, ten repeated values | 0.05466 → 0 |

F32 centroid memory is now measured at 16 bytes versus 12 before the weight fix; F64 stays 24 bytes. At 142 centroids this adds 568 bytes (+31.6%) to the F32 logical digest size. Compact modern state is larger than compact original state for many tiny/continuous digests. Original result compaction reduced the 10M-input f64 reserved vector capacity from 120,000,000 to 1,776 bytes with identical centroids; capacity is **not RSS savings**.

The largest matrix input is 10M unit samples, below 2²⁴. Separate regression tests cover the corrected large-pile accounting; this benchmark does not establish arbitrary exact mass conservation. Subnormal masses and overflowing totals remain separate open limits. Modern improvements are workload-dependent; small/sorted builds and repeated pairwise merging can lose.

## Preserved historical evidence

The pre-weight-fix committed run remains in [RESULTS-756d221.md](RESULTS-756d221.md), [raw data](results/2026-10-14-756d221.jsonl) and [metadata](results/2026-10-14-756d221-metadata.json). It measured 12-byte F32 centroids and must not be presented as current performance. The new [comparison](results/2026-10-14-22867bd-comparison.json) shows that all recorded numerical fields, centroid counts and wire lengths are unchanged; only 13 F32 logical-memory fields differ. Cross-run timing medians are descriptive rather than paired evidence of a code speedup or regression.

The earlier intermediate run remains in [RESULTS.md](RESULTS.md), [raw data](results/2026-10-14.jsonl) and [metadata](results/2026-10-14-metadata.json). It used `b89527187219c83768f50416fa8ac61fe3ca2f6d` plus the [measured core patch](measured-core.patch), before later interpolation and wire fixes. Cross-run timing differences alone do not establish a code performance change.

## Reproduce on the current checkout

Run on Linux or WSL with Rust and Python 3.11+ installed. Use a stable checkout with no concurrent source changes or heavy workloads. The harness needs the normal repository dependencies already available in Cargo's cache (`--offline` is intentional).

```sh
# Independent original source; its working-tree contents are not trusted or used.
git clone https://github.com/G-Research/tdigest-rs.git target/compat-performance/upstream

# Fast validation before the full matrix. The runner checks the original git
# blobs against the pinned normalized source hashes before compiling anything.
python3 compat/performance/run.py \
  --upstream target/compat-performance/upstream --smoke

# Full matrix: 30 cases, including up to 10M inputs, seven distributions,
# weighted data, scalar queries, and ~40-way merges. May take several minutes.
python3 compat/performance/run.py \
  --upstream target/compat-performance/upstream
```

Results go under `target/compat-performance/run-<UTC timestamp>`; the runner refuses to overwrite a nonempty directory. Use `--output`, `--cpu`, or `--jobs` to select these explicitly. It records Git status, exact core hashes, compiler, CPU, binary/harness hashes, environment overrides and timing samples, and rejects source changes during compilation or execution. The default Cargo target directory is `target/compat-performance/cargo`; set `CARGO_TARGET_DIR` to reuse a suitable existing release cache. Dependencies and build profiles must match to benefit from that cache. No benchmark updates the Python development environment.

Compare two completed runs with the same harness, original pin, compiler and
environment settings using the read-only helper:

```sh
python3 compat/performance/compare.py \
  --baseline compat/performance/results/2026-10-14-756d221.jsonl \
  --baseline-metadata compat/performance/results/2026-10-14-756d221-metadata.json \
  --candidate compat/performance/results/2026-10-14-22867bd.jsonl \
  --candidate-metadata compat/performance/results/2026-10-14-22867bd-metadata.json \
  --output target/compat-performance/repeated-comparison.json
```

The helper refuses to overwrite its output and reports full non-timing changes,
original numerical controls, timing medians/ranges and inferred logical layouts.
It does not make a statistical speedup claim.

The original Rust files are extracted from the requested commit into an isolated small crate; its Python workspace member is omitted. The harness links this independent original crate alongside the real modern crate. Both use the same process allocator. Python, Java and Polars overhead is excluded.

## Replay the historical measured core

For source-level replay, create a separate checkout at `b89527187219c83768f50416fa8ac61fe3ca2f6d`, apply `measured-core.patch` there, and copy this `compat/performance` directory into that checkout. Run the same commands above against an independent upstream checkout. Do not apply the historical patch to the current development branch: it intentionally describes an earlier intermediate core. Compiler/toolchain, CPU, allocator, background load and thermal differences can still change timings; source replay does not promise identical machine code or timing samples.

## How to interpret accuracy and speed

- Quantiles use a 1001-point probability grid. Rank error is distance outside the empirical interval `[fraction below estimate, fraction at or below estimate]`; ties are permitted. This is a sampled maximum, not a proven worst-case bound.
- Value errors use type-7 linear quantiles as a disclosed reference. Different interpolation conventions can disagree on tiny/discrete samples. Trimmed-mean references use exact fractional-mass trimming. CDF uses midpoint empirical ranks at 257 probes.
- Same delta does not imply equal centroid count or memory. Counts and logical memory are recorded. The original core's reserved capacity is recorded separately.
- A sequential left fold is compared with another sequential left fold. Modern one-shot k-way merge is a distinct option with separately recorded accuracy; its timing ratio is not a claim of unchanged numerical behavior.
- One machine, one seed per distribution, no competitor libraries, and no confidence intervals: small timing differences may be noise. Wider competitiveness needs multiple seeds, machines, matched memory/accuracy budgets, representative competitors, and production-shaped traces.

Useful next experiments are scalar-CDF allocation reduction, profiling the modern per-candidate logarithmic compression limit, and reducing tiny-digest memory. Keep any new numbers and behavioral changes explicit, and rerun after correctness fixes before drawing release conclusions.
