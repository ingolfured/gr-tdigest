# Proposed accuracy study under common logical storage caps

Status: **fresh corrected-source protocol; static preparation only, no experiment executed**. The expanded delta grid and all three illustrative error thresholds were agreed before the policy freeze or any new training. This is separate from the completed delta-300 survey. The existing survey, scripts, raw data and interpretation remain unchanged.

## Question and scope

Compare the original and opt-in modern cores under the same declared **logical retained storage cap**. This is not a claim of equal actual memory, equal allocation, equal RSS, optimal tuning, speed, or global accuracy leadership. Keep original-compatible API behavior separate from the opt-in modern API comparison.

Use immutable modern commit `ff73cfab31d6368d203c1a361b8efaaea29d735b` and independent original `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Reuse the exact validated Rust generator, reference helpers, query rounding and public wire decoder from the completed survey. Neither input repository is edited. The proposed execution runner must export and hash both sources, retain the built executable/Cargo.lock/toolchain, and freeze its own complete protocol before a new smoke. Old smoke evidence is useful history, not sufficient authorization for the new delta grid.

## Predeclared settings

- Delta grid for both cores: **25, 50, 100, 150, 200, 300, 600, 1,200**. Modern K2Norm / singleton Off. The upper settings prevent the 4 KiB comparison from being artificially limited by an old delta-300 grid ceiling.
- Logical caps: **1,024, 2,048, 4,096 bytes per final digest**. F32 and F64 means evaluated separately.
- Separate illustrative maximum queried-rank thresholds: **0.005, 0.01, 0.02** (0.5, 1.0, 2.0 percentage points). These are experiment thresholds, **not product tolerances**.
- Training seeds: **1009, 10007, 65537**.
- Fresh held-out seeds: **15485863, 32452843, 49979687**. These differ from every seed in the completed survey and its reserved training set.
- Distributions: uniform, normal, exponential, lognormal sigma 3, Pareto alpha 1.5, ten-point ties, 90% zero plus lognormal tail, narrow modes near +/-10.
- Sizes: 32, 1,000, 100,000; all input weights are one. Existing fractional-weight audits remain separate.
- Direct construction and 40-shard balanced-IID, sorted-range and 90%-in-one-shard layouts. Original sequential merges; modern sequential and k-way merges. Reverse shard order additionally at n=100,000. Empty shards skipped equally and recorded.

Training runs both cores at each grid delta, using identical inputs. This is **1,152 process invocations / 16,128 state records** (10,368 modern / 5,760 original). There is no distribution-specific, size-specific, seed-specific or route-specific delta selection.

## Selection rule, fixed before viewing training data

For each `(core, mean precision, logical cap)`, compute the maximum logical retained bytes over **all** training seeds, distributions, sizes and supported routes at each candidate delta. A candidate is feasible only if every declared training state fits the cap and execution/invariant gates completed. Select the **largest feasible grid delta**. The rule does not assume storage or accuracy is monotone: it checks every candidate independently.

The storage policy uses training storage, not training error, to choose the compression setting. It avoids an additional subjective multi-metric winner rule, but does **not** promise the best accuracy achievable under the cap. Retain and report every grid point's accuracy and storage, including cases where a lower delta happens to be more accurate. Record the selected delta, maximum/median training bytes, unused cap, and whether selection reaches the grid's upper edge. In particular, disclose remaining slack when the largest grid delta is selected. Do not interpolate/extrapolate to a missing delta.

Independently, for each `(core, mean precision, illustrative rank threshold)`, choose the **smallest grid delta** whose maximum queried-rank error over every training dataset and supported route is at most that threshold. Any nonfinite query error makes that candidate ineligible for this error policy, while its raw observations remain recorded. This is a minimum-delta policy, not a claim of minimum actual bytes. It does not simultaneously enforce the storage caps. Record all candidates, including nonmonotone error behavior; do not blend rank thresholds with value/CDF objectives. Freeze these error-policy settings alongside the storage selections.

The original's n=32 endpoint behavior previously produced about 1.55 percentage points maximum rank error. The 0.5/1.0-point policies may therefore be unavailable for the original at every delta. Keep that honest outcome. The separately predeclared 2.0-point threshold may permit a common comparison above that floor; it does not replace or loosen the other two policies. No tiny cases are excluded.

Modern k-way storage is included in its training feasibility check because the resulting setting will be used across all advertised modern routes. The original has no k-way route; same-route accuracy comparisons therefore use only direct/sequential routes. Modern k-way is an explicitly separate comparison to original sequential merging, not an algorithmically identical route.

If no candidate fits a core/precision/cap or passes an error threshold, mark it **unavailable within this grid** and retain all training data. Do not call the smallest over-cap candidate a budgeted result. If only one core is feasible, test the available core on held-out data without claiming a common-budget/error comparison. The reused frozen two-core harness may execute the unavailable companion at delta 25, solely as a clearly tagged diagnostic to satisfy its fixed CLI; retain that output but exclude it from all policy claims. If both are unavailable, skip that cell and record why. A future broader grid requires a new protocol and new held-out seeds.

Write and hash a `selection-lock.json` containing all **24 decisions** (12 storage and 12 illustrative error-policy decisions), the source protocol hash and all training raw-file hashes. Held-out execution must require that existing lock and reject edits, incomplete training, failed modern invariants, unexpected coverage or any source/protocol mismatch.

## Held-out execution and budget misses

Use exactly the selected global delta for each core/precision/policy across all fresh held-out inputs and routes. Across three storage and three error policies there are at most **864 invocations / 12,096 state records** (7,776 modern / 4,320 original, including any explicitly diagnostic companion records). The same selected pair may recur at different policies; the first implementation repeats and labels the pairs, with summaries stratified by policy. Do not silently multiply duplicate results as independent evidence or pool different budgets into a single winner count.

A selected setting can exceed its training storage cap or error threshold on new data. **Keep those states and their errors, flag the violation, and do not change delta.** Report all fixed-policy results first, with violation counts and worst excess bytes/rank error. A secondary subset of states within the cap must be labelled conditional; never present that subset alone. If a cap or threshold is exceeded, the protocol has evaluated a training-selected policy, not established a guaranteed memory/error bound. Value errors and CDF errors remain visible per distribution even when a rank threshold passes.

## Metrics and conventions

Use the same post-cast sample and the same effective query probabilities for both cores (including F32 probability rounding). Query levels remain 0..1 by .001 plus .0001 and .9999. No new queries are selected after seeing held-out failures.

- Quantile rank error: distance to the tie interval `[F(x-), F(x)]`. Report p50, all-query maxima, and tails `q <= .01` / `q >= .99` separately.
- Value error: absolute Type-7 error at p50 and the same tail grid. Also report p50 error / sample IQR where nonzero; zero-IQR cases stay visible with absolute errors and an undefined normalized value. Rank error and value error can disagree sharply in separated modes.
- Modern CDF: absolute error against midpoint ECDF at the already fixed empirical/atom/adjacent/gap probes, including exact atom and adjacent-value behavior. The original has no CDF API; no synthetic old CDF is supplied.
- Per-state storage: object bytes, centroid count and payload bytes, cap slack/excess. Modern vector capacity is not publicly available: mark it unknown. Original compacted vector capacity is recorded separately. Neither is process RSS or peak merge memory.
- Same-source wire query parity, finite/bounded/monotone modern queries, centroid validity, count/mass and extrema invariants remain mandatory. Keep all inherited original quirks as observations, not a reason to censor cases.

Report per distribution, size, precision, cap, route and seed before any aggregate. Aggregates give each declared state equal weight within a cap and explicitly distinguish original/modern route populations. Use descriptive median, nearest-rank p95 and worst errors; three seeds do not establish reliable extreme-tail probabilities. Report query/centroid drift from original separately from error against the sample. Do not call a positive drift a correctness failure merely because the original differs.

## Gates and limits

Before the sweep, reuse the independent rational oracle's 14 hand-computed assertions and seven Rust reference fixtures / 763 numeric cells. Then run a new grid smoke: eight distributions, F32/F64, n=32, seed 1, all eight deltas = **128 invocations / 1,408 states**. Independently validate every tiny reference and inspect invariants, source hashes and complete coverage. The successful smoke and protocol must precede training, and the training selection lock must precede held-out execution.

Proposed hard execution limits, separate from the digest storage caps:

| Phase | Dataset runtime | Compressed raw output cap |
|---|---:|---:|
| Grid smoke | 180 s | 128 MiB |
| Training | 720 s | 1,152 MiB |
| Held-out | 600 s | 768 MiB |

Additionally enforce a **1,200-second (20-minute) global dataset-execution budget** across all phases and a **2 GiB global compressed raw cap**. The phase ceilings do not override the stricter global remaining budget. Build timeout is separately 1,800 s, at most two Cargo jobs, only during an assigned CPU/cache slot. Each case has a 120 s timeout; wall budget is checked between cases, and each case timer uses the smaller of 120 s and the global/phase remaining runtime, so the global runtime cannot gain an extra full case. Streaming output has an 8 MiB maximum line length. Compressed-size limits are checked after every state flush; maximum overshoot is one allowed state line. Keep partial gzip files/logs and mark budget-limited output incomplete, never successful. No automatic resume or overwrite. These limits need to be implemented and smoke-tested in the new runner before execution.

The prior run suggests roughly 11 minutes of dataset execution and 1.3 GB compressed output for training plus held-out, but grid delta affects both costs. This is a planning estimate, not a benchmark or completion promise. Additional allocations/source exports/logs are outside the compressed raw cap and must be reported separately. No storage capacity or memory-pressure guarantee is implied by these disk limits.

## Prepared files and remaining work

`storage_budget_plan.py` describes the static case matrix and selection policy without calling the library. `select_storage_budget.py` is a draft validator/selector that streams a complete future training run and writes a selection lock. Both are new files; they do not modify the completed study. The four frozen reference/execution source files are copied alongside them for audit.

The execution driver is prepared and source-bound, but has not built or run this study.
It freezes sweep-aware filenames, policy selections, diagnostic companions,
budget-violation reporting, source manifests and global execution/output limits.
Its own hash and the policy hash are retained in `execution-protocol.prepared.json`.
The synthetic driver tests exercise guard behavior without importing either core.
Parent review and successful full-source gates must precede the fresh grid smoke;
the smoke must pass before training, and a complete training-only selection lock
must precede held-out execution. No build, core call, oracle execution or dataset
generation has occurred for this new revision.

## Revision provenance and package identities

This separate directory preserves the unexecuted historical 52f6212 protocol.
The modern source revision now includes the reviewed K2Norm nonpositive-domain
fix; all scientific settings, seeds, grid points, routes, selection rules and
hard budgets remain unchanged. The four earlier reference/generator files stay
byte-identical. Their historical source constants are reference provenance only;
`storage_budget_plan.py` and the source-bound new driver govern this study.

The modern workspace now includes the public original `tdigest_core` package.
To avoid two distinct path packages with that same name/version, the independently
exported original is named `tdigest-original-reference` in its harness-only Cargo
manifest. The harness dependency retains alias `tdigest_core` with an explicit
`package` attribute. No original Rust source or arithmetic is renamed or edited.
This packaging identity is captured in the execution protocol and source manifest.
The generated harness resolves offline from the copied modern lock; its final
lock and toolchain are recorded separately. This is not the committed-original-lock
oracle and must not be described as that provenance.

A later selected-budget timing study is separate. It may reuse this study's
globally selected parameters on fresh seed 86028121 at n=1,000,000, but these
parameters were trained only through n=100,000. Any storage/rank budget misses
on that larger workload stay visible; selection is not a universal guarantee.
No timing binary, timing result or selected parameter is created here.
