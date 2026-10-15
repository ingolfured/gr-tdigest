# Small timing follow-up at independently selected budgets

Status: review draft with scratch executor and synthetic guard tests implemented. No native compilation, data generation or measurement has run. The machine's current CPU/cache assignment remains unchanged. `PROTOCOL.draft.json` is the structured counterpart to this plan. Modern source is fixed at `ff73cfab31d6368d203c1a361b8efaaea29d735b`, corrected accuracy protocol SHA256 `9c2a455e689802eb2dbbe7764764238b794fa5975e2517d5e3c1355fed66904a`. The selection-lock hash remains a mandatory future runtime input. Implementation review and a source-bound native smoke still precede full execution.

## Question and prerequisite

The new question is whether build, merge and query speed differs at compression settings selected independently for logical storage or illustrative rank error, and what actual accuracy/storage accompanies that speed on fresh input. Earlier fixed-delta timing does not answer this. This is a small follow-up, not a repeat of the large 30-case comparison.

First complete the corrected-source accuracy study and its reviewed training/held-out gates. Reuse its `selection-lock.json` unchanged. Both the modern source commit and every generator/reference file must match that corrected protocol; the historical unexecuted 52f6212 storage protocol is ineligible. The independent original remains `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Timing results must never select a new delta or change an existing accuracy policy.

## Fixed small matrix

- One fresh seed: **86028121**, distinct from the predeclared training and held-out seeds.
- **1,000,000 unit-weight inputs** per dataset: normal, ten-point ties, and narrow separated modes near +/-10, using the frozen generator and post-cast samples.
- F32 and F64 means, separately.
- Two storage policies: **1,024 and 4,096 logical bytes**. One separate illustrative error policy: **0.02 maximum queried rank error** (2 percentage points). The latter is an experiment threshold, not a product tolerance.
- Use each core's global training-selected delta for that precision/policy. This yields at most **18 paired cells**. Report all 24 selection decisions, including unavailable 2 KiB and 0.5/1-point policies that are outside this timing subset.
- If either core has no selected delta for a policy, report both selection outcomes and omit that paired timing. Do not substitute delta 25 or another setting. Any existing one-core accuracy evidence remains visible separately.
- Before timing, deduplicate identical `(distribution, n, seed, precision, original_delta, modern_delta)` pairs and attach their policy labels. Reused timings are not independent evidence.

Each dataset is also shuffled with the frozen generator's shuffle rule and divided round-robin into **40 balanced IID shards**, 25,000 values each. Use fixed forward shard order. This follow-up deliberately omits range/skew/reversed merge layouts; the separate accuracy survey retains those. No fresh seed or distribution is chosen after observing timing or accuracy.

## Native operations and fair timing boundaries

One immutable release executable links both the modern source and an independent original crate. It uses the same Linux GNU jemalloc allocator and dependency graph for both core modes. Keep the reference crate's package name distinct from the restored public `tdigest_core` dependency; source arithmetic bytes must still match the original pin. This is native Rust evidence, not Python adapter timing or a standalone original/system-allocator comparison.

1. **Build from an immutable input slice.** Original `from_array` includes its internal copy/sort. Modern `values.to_vec()` plus builder/`merge_unsorted` runs inside the timer because its API consumes a vector. Input generation, reference sorting, state inspection and output destruction are outside the timer. Use one warmup and three single-build samples, with no calibration or adaptive repetition.
2. **40-way sequential merge.** Prebuild the 40 leaves at the selected core delta outside timing. Inside timing, clone the first leaf and apply the other 39 merges through each public API. All required reset/ownership copies and allocations are included; leaf construction is not. Use two warmups, then five batches of eight complete reductions.
3. **Modern 40-way k-way merge.** Use the same modern leaves. Include cloning/collecting the owned leaf vector and `merge_digests` inside timing. The same warmups/batches apply. Compare with modern sequential, and label any comparison with original sequential as a different algorithmic route. The original has no equivalent k-way API.
4. **p50 queries** on each resulting direct/sequential/modern-k-way state. Use two warmup batches and five batches of 4,096 calls, passing input and result through `black_box` every call. Report per-call time. Do not assume build and merge states have identical accuracy.
5. **Modern scalar CDF** on those states, absolute time only. Cycle equally through post-cast Type-7 sample values at effective q=.1/.5/.9: 4,095 calls per batch, two warmup and five measured batches. Each call uses the public one-element CDF API, including its result allocation/destruction. No old CDF estimate or ratio is invented.

For merge microbatches, allocate the result-holder capacity outside timing, retain all eight results until the timer stops, and then validate/drop them. This makes the stated output-destruction exclusion real rather than accidentally timing seven of eight drops. `black_box` inputs/results and use no I/O, JSON, hashing or reference calculation inside measured intervals. Raw batch elapsed times and iteration counts are retained. Sub-resolution results stay labelled; do not add repetitions after seeing them.

## Counterbalancing and controls

Sort deduplicated cases by the declared policy/distribution/precision order. Block 1 visits forward and runs original then modern in adjacent fresh processes. Block 2 visits the reverse order and runs modern then original. This is at most **72 native processes** and two independent process observations per core/cell. Within each process, block 1 runs build, sequential, k-way if modern, p50, CDF if modern; block 2 reverses that order. Canonical states needed for queries are prepared before the measured sequence.

Use the exact same executable for both core modes. Require CPU **0** affinity and `RAYON_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`; fail if CPU0 is unavailable rather than choosing another CPU silently. The parent must release the sole CPU/cache slot first. No other build, benchmark, heavy audit or installation runs concurrently. Record toolchain/target/profile, source manifests, Cargo.lock, harness/runner/protocol/selection hashes, binary hash, environment, CPU information, affinity, process schedule and available frequency/load observations. Do not alter governor settings.

Compare adjacent original/modern timings within each block, report both block ratios and their range, and show each core's block-2/block-1 drift as a control. Opposing block results or a similar original-control shift limit attribution. Two blocks support descriptive evidence, not a significance claim. Do not combine policy-labelled duplicates, routes or distributions into a headline winner count.

## Accuracy, storage and invariant record for every cell

Outside timers, reuse the corrected study's same 1,003 effective quantile levels, Type-7 reference, weighted-tie rank interval convention, and empirical/atom/adjacent/gap CDF probes. Record p50 rank/value error and value/IQR where defined, overall and tail rank/value maxima, and modern CDF absolute errors separately by route. Preserve original inherited out-of-range/nonmonotone observations; modern finite/bounded/monotone, mass, centroid and wire-parity failures invalidate the run rather than disappearing from its data.

Record the actual final centroid count, object bytes, centroid payload bytes and logical retained bytes, plus the 40 leaves' counts/logical sizes. Record original vector capacities from the actual timed states without compacting them; modern capacity is unknown through the public interface. Wire bytes are a separate modern field. These quantities are not peak merge memory, equal allocation or equal RSS, and the original compatibility adapter's capacity compaction is not timed here.

For each storage-policy label, record final cap slack/excess and a violation flag for every route. For the error-policy label, record fresh-input maximum-rank threshold excess and a violation flag. At n=1M these are new observations beyond the accuracy study's largest n=100K: training feasibility is no guarantee. Keep misses in the main timing/error table and never retune. Say 'training-selected policy, violated on this input' where appropriate, rather than 'speed at matched error/storage'. Only cells in which both actually meet a declared bound can support that qualified comparison, and they must be shown alongside all misses. CDF/value error are not constrained by the rank policy.

## Freeze, smoke, budget and output

Before running: supply the future selection hash, validate complete immutable source and mandatory-file manifests, independently review the scratch implementation against this protocol, and hash it all. The implemented driver checks all completed corrected-source accuracy phases, the held-out training-only selector replay, its locked choices and exact original/modern exports. It copies the original resolved dependency lock and keeps the same standalone package identity so the new harness builds with `--locked`. Verify the same artifacts before and after each block. Prior timing executables are not substitutes for a new source/harness-bound build.

First run a bounded **n=32 smoke** covering the selected distinct parameters, all three distributions, both precisions and both cores (at most 36 process invocations). Reuse exact tiny-reference checks, validate timing-route state fingerprints against the accuracy helpers, check output/drop boundaries and source/binary/control hashes, and prove stopped/incomplete handling with synthetic child processes. Smoke timings make no performance claim. Native full execution requires explicit parent acceptance of the completed protocol and smoke.

The full timing driver has a hard **600-second wall limit**, including fresh input/reference construction, state checking, warmups and measured calls. Each child gets at most 30 seconds or the remaining global budget, whichever is smaller. Smoke has a separate 60-second ceiling. Retained raw timing/metric output is capped at 64 MiB. Build/preflight is outside those timing budgets, separately capped at 1,800 seconds with at most two build jobs during the assigned CPU/cache slot. Expect a few minutes of full execution, but completion is not assumed or promised. If a cap is reached, stop, retain partial logs/results and mark incomplete; no automatic resume, repetition change or censored successful subset.

The report should be a compact per-cell table with both selected deltas, both block timings/ratios, state bytes, p50/tail errors and policy violations, plus separate modern k-way/CDF rows. It must state the seed, source/selection hashes and untested layouts/platforms. This experiment cannot establish world-best performance, universal bounds, equal RSS, or equal error merely because training selection succeeded.
