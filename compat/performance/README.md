# Linux performance and accuracy evidence

The compatibility interface (`tdigest_rs`) keeps the original arithmetic. The modern engine (`gr_tdigest`) is opt-in. The useful story is measured tradeoffs: compatibility, much smaller reserved result capacity, and workload-dependent modern speed/accuracy improvements. These measurements do **not** establish a global speed or accuracy lead.

## What was measured

The [recorded run](results/2026-10-14-metadata.json) took place on 2026-10-14, 21:02–21:05 UTC: release Rust cores, Linux/WSL, Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0, delta=300, f32 and f64. Each timing uses calibration and seven batches; full batch samples and accuracy results are in [raw JSONL](results/2026-10-14.jsonl). The original core comes from independently extracted G-Research commit `8d9cf1fe82c5c9254e5dc9593f01717e317f4333` with [verified checksums](original-source-provenance.json).

**This is a historical intermediate snapshot**, based on `b89527187219c83768f50416fa8ac61fe3ca2f6d` plus first-wave CDF, quantile and extrema fixes. [The exact measured core patch](measured-core.patch) and per-file hashes identify it. Later overflow-safe interpolation and wire-validation fixes were **not** in this binary. Do not describe these numbers as a fresh benchmark of a later commit. The runner can produce new evidence against the current checkout.

Selected results from this one run:

| Workload | Result |
|---|---|
| Build 10M random uniform f64 values | Original 3.215 s; modern 0.808 s (~4.0× faster) |
| Build 10M random uniform f32 values | Original 2.473 s; modern 0.529 s (~4.7× faster) |
| Build 1K random uniform f64 values | Original 18.2 µs; modern 21.2 µs (modern slower) |
| Build 100K sorted f32 values | Original 787 µs; modern 1,120 µs (modern slower) |
| Merge 40 random uniform f64 shards, 100K total values | Modern sequential fold is slower; modern one-shot k-way merge is ~1.5× faster than original sequential fold, with different recompression semantics |
| Quantile grid on 100K values from ten repeated values | Maximum sampled rank error 0.05466 original → 0 modern; 142 → 10 centroids |
| 10M uniform f32 quantile accuracy | Maximum sampled rank error 0.002082 original → 0.0000806 modern |
| Original-core f64 result after 10M inputs | Reserved vector capacity 120,000,000 bytes → 1,776 after `shrink_to_fit`; identical centroid values and weights |

Compaction is a numerically neutral operation used by the compatibility adapter; the harness measures that operation separately from the independent original core. Capacity bytes are **not RSS savings**. The modern object and centroid layout also consume more logical bytes than the compact original representation for many tiny/continuous digests. See the full [results and limitations](RESULTS.md) before selecting a default or making production claims.

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
