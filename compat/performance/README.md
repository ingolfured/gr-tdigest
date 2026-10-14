# Linux performance and accuracy evidence

The compatibility interface (`tdigest_rs`) keeps the original arithmetic. The modern engine (`gr_tdigest`) is opt-in. The useful story is measured tradeoffs: compatibility, much smaller reserved result capacity, and workload-dependent modern speed/accuracy improvements. These measurements do **not** establish a global speed or accuracy lead.

## Current committed evidence

The latest full run targets clean commit `756d221d35bd6d382cdfb8705598bb445a665997`; no source changed during the run. See [current results and limitations](RESULTS-756d221.md), [raw samples](results/2026-10-14-756d221.jsonl) and [provenance](results/2026-10-14-756d221-metadata.json). It includes the first-batch correctness fixes. The scratch scalar-CDF optimization is not included.

Release Rust cores on Linux/WSL, Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0, delta=300, both precisions, seven batches per operation:

| Selected workload | Original → modern |
|---|---|
| Build 10M random uniform f64 values | 2.896 s → 0.634 s (~4.6× faster) |
| Build 10M random uniform f32 values | 2.186 s → 0.427 s (~5.1× faster) |
| Build 1K random uniform f64 values | 15.3 µs → 18.6 µs |
| Build 100K sorted f32 values | 1121 µs → 1429 µs |
| Merge 40 uniform f64 shards, 100K total values | Pairwise modern is slower; modern one-shot k-way is ~1.5× faster than original pairwise |
| Maximum sampled rank error, ten repeated values | 0.05466 → 0 |

The independent original core plus numerically neutral compaction reduced 10M-input f64 reserved vector capacity from 120,000,000 to 1,776 bytes with identical centroid contents. Capacity is **not RSS savings**. Compact modern state is also larger than compact original state for many tiny/continuous digests.

**Known unresolved limit:** modern f32 weight/mass accounting beyond this matrix remains open. The largest input here is 10M unit samples, below 2²⁴; these results do not prove exact mass conservation for larger or arbitrarily scaled/fractional f32 weights.

## Preserved historical evidence

The earlier intermediate run remains in [RESULTS.md](RESULTS.md), its [raw data](results/2026-10-14.jsonl), and [metadata](results/2026-10-14-metadata.json). It used `b89527187219c83768f50416fa8ac61fe3ca2f6d` plus the [measured core patch](measured-core.patch), before later interpolation and wire fixes. It must not be relabeled as current evidence. 14/30 cases had identical non-timing fields between the runs. The other cases show tiny modern rounding differences after overflow-safe interpolation: the largest stored quantile change is 1.1921e-7 near ±1e9; original outputs, centroid counts and capacities are unchanged. See the full comparison in the current report. Timing differences alone are not evidence of a performance change.

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
