# Linux performance and accuracy evidence

The [controlled interleaved follow-up](RESULTS-interleaved-20261015.md) compares
immutable `22867bd` and `c5dec35` with 32 counterbalanced processes. Exact states
and complete modern wire bytes match. The earlier double-digit large-case
slowdown did not reproduce consistently across blocks; original-control shifts
also prevent calling the observed p50 movement a new modern speedup. Small
sorted-case differences remain descriptive. This does not prove zero overhead
or measure the newer transactional-scaling/packaging source. Raw results and
[reproduction details](interleaved/README.md) are retained with their source pins.


The compatibility interface (`tdigest_rs`) keeps the original arithmetic. The modern engine (`gr_tdigest`) is opt-in. The useful story is measured tradeoffs: compatibility, much smaller reserved result capacity, and workload-dependent modern speed/accuracy improvements. These measurements do **not** establish a global speed or accuracy lead.

## Current committed evidence

The latest unchanged 30-case matrix targets clean candidate `c5dec35059990b40fe6f036dc83590bf0cdf2a70`, including finite-mean bounds and subnormal-total query normalization. Source hashes were stable throughout compilation/timing. See [current results](RESULTS-c5dec35.md), [raw batches](results/2026-10-14-c5dec35.jsonl), [provenance](results/2026-10-14-c5dec35-metadata.json), and [comparison with 22867bd](results/2026-10-14-c5dec35-comparison.json). The transactional scaling and packaging-hygiene proposals are outside this measured source.

| Selected workload | Original → modern |
|---|---|
| Build 10M uniform F64 values | 2.878 s → 0.789 s (~3.6× faster) |
| Build 10M uniform F32 values | 2.252 s → 0.487 s (~4.6× faster) |
| Build 1K uniform F64 values | 15.7 µs → 18.8 µs |
| Build 100K sorted F32 values | 598 µs → 969 µs |
| Merge 40 F64 shards, 100K uniform values | Original sequential 325.0 µs; modern sequential 555.7 µs; modern k-way 204.7 µs |
| Maximum sampled rank error, ten repeated values | 0.05466 → 0 |

Linux/WSL, Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0 release, delta 300, seven timed batches per operation. Results are workload-specific; same delta does not mean same memory. F32/F64 centroid sizes remain 16/24 bytes. Modern state is larger than compact original state for many tiny/continuous digests; capacity is not RSS. No global speed/accuracy lead is established.

Against 22867bd, 0 of 30 cases changed non-timing fields; see the complete comparison rather than treating cross-run timing variation as a proven code effect. Separate [validation](../validation/c5dec35-validation.json) covers the corrected numerical boundary cases and installed wheel.

## Preserved historical evidence

The prior [22867bd report](RESULTS-22867bd.md) remains intact. It predates the finite-mean and subnormal-query fixes, while retaining the same f64-weight representation. The new c5dec35 run has identical non-timing fields across all 30 cases. Large-build and p50 timing differences need a focused interleaved comparison before attributing them to code.

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
