# t-digest experiments and continuation handoff

Prepared 18 October 2026 from saved local research. This note does not refresh GitHub status, publish branches, or certify that local artifacts have been backed up. The accompanying branch/archive inventory should establish that separately.

## Current position

The 15 October experiment campaign completed **20 prototype dispositions: one retained, 19 rejected under the original strict performance/quality gate**. Rejected candidates stopped at decisive failures; this does **not** mean all 20 completed full heldout and release validation.

The retained change is **E12, personal [PR #9](https://github.com/ingolfured/gr-tdigest/pull/9)**: copying numerical output slices directly into NumPy-owned arrays instead of creating Python scalar/list intermediates. **Experiment E09 is a different, rejected count-only compression-cap search prototype. E09 is not PR #9.**

The user subsequently relaxed the per-microbenchmark veto: accept disclosed local performance costs when representative end-to-end workloads benefit materially. Production uses large servers, often more than 100 CPUs, with abundant RAM. The objective is increasing throughput with available cores. Physical cores, SMT, CPU quotas and NUMA topology are still unspecified. This preference was recorded on 16 October; it did not itself execute new experiments or turn rejected prototypes into validated winners.

Historical review order is **[PR #6](https://github.com/ingolfured/gr-tdigest/pull/6) → [#7](https://github.com/ingolfured/gr-tdigest/pull/7) → [#8](https://github.com/ingolfured/gr-tdigest/pull/8) → [#9](https://github.com/ingolfured/gr-tdigest/pull/9)**. The saved 15 October state leaves [G-Research/tdigest-rs PR #38](https://github.com/G-Research/tdigest-rs/pull/38) draft, with PR #9 not incorporated. Use the fresh remote inventory for present status.

## Compatibility boundary and provenance

- Keep the original public `tdigest_rs.TDigest` API and original arithmetic. The modern `gr_tdigest` engine is explicit opt-in. The original API must not require users to adopt a `LegacyTDigest` name.
- Experiment baseline: `25c8c1642029a640b153c160c22a89795d29ad91`; recorded modern-source reference: `ff73cfab31d6368d203c1a361b8efaaea29d735b`.
- Independent original upstream oracle: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Preserve its expected data and untouched original suite. Saved final coverage includes 35 unchanged original source files, 91 original-suite tests, and 70 oracle cases with 1,762 exact-bit checks plus 126 NaN-class checks in each debug/release profile.
- Builds, tests and benchmarks run on Linux/WSL, originally ArchFresh as user `ingo`. The isolated checkout was `/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest`. Do not modify or clean the dirty original `/home/ingo/git/gr-tdigest`.
- Research root: `/var/tmp/codex-tdigest-pr3-20261014/day20261015`. Windows research root: `C:/Users/ingol/Documents/Codex/2026-10-14/hi/work`. Paths below identify original artifacts; package their contents before retiring this computer. A path string or source branch alone is not an evidence backup.

## All 20 historical decisions

Percentages describe the recorded elapsed-time comparison for the stated operation, not universal speedups. The original gate treated any certified regression as a rejection. See the evidence index below for full cells, uncertainty, skipped work and provenance.

| ID | Prototype | Original decision and decisive evidence |
| --- | --- | --- |
| E01 | Preserve mass at tiny `Use` caps | Rejected: mass repaired but rank/CDF errors worsened at matched storage. For `[0,1,2,100]`, cap 2 median changed from 0 to 25.75 versus exact 1.5. [Q] |
| E02 | Borrow operands instead of cloning for merge | Rejected: fresh p50 control 9.71% slower; 294 confirmation cells skipped. Earlier ties-only merge gains of 4.20–7.08% remain recorded. [N2] |
| E03 | Bypass heap for two nonempty runs | Rejected: five fresh p50 controls 3.75–6.71% slower. Strong two-way merge gains remain recorded. Now a priority for renewed evaluation. [N38] |
| E04 | Scalar CDF without prefix allocation | Rejected: unchanged batch-CDF control 7.33% slower; all 42 scalar target cells short/inconclusive. [NB] |
| E05 | Move compressor vectors through no-edge paths | Rejected: singleton build/drop 13.23% slower; no qualified gain in 16 lifecycle cells, despite allocation reductions. [L] |
| E06 | Retain positive residual bucket tail | Rejected: retained moment improved, but rank/value/CDF errors worsened. Far-tail p90 changed from 1 to `6e16` versus exact 1. [Q] |
| E07 | Hoist scale constants | Rejected: build case 7.24% slower; no qualified gains in the 24-cell screen. [B] |
| E08 | Replace heap root during k-way merge | Rejected: three fresh batch-CDF controls 4.37–5.02% slower. Earlier k-way gains remain recorded. Now a priority for renewed evaluation. [N38] |
| E09 | Count-only cap search before materializing | Rejected: two p50 controls 5.79–6.64% slower, despite seven capped sequential-merge gains and fewer allocations. This is **not PR #9**. [N] |
| E10 | Consume first owned frontend merge input | Rejected: five lifecycle regressions, four gains in 54 cells; benefits mostly singleton cases. Final capacity repair and memory gates passed. [L] |
| E11 | Contiguous NumPy input fast path | Rejected: three small strided-input cases 4.77–6.81% slower. [W] |
| E12 | Bulk-copy NumPy outputs | **Retained as PR #9.** Exact values/ownership preserved; installed-wheel and compatibility gates passed. No core numerical algorithm change. [R] |
| E13 | Encode directly into Python bytes | Rejected: ten native Rust encode cases 6.78–11.60% slower despite six Python encode gains of 42.87–57.28%. Conditional priority for Python serialization workloads. [C] |
| E14 | Decode/validate Polars input once | Rejected: no qualified gain after full screen and registered longer repeat; nine neutral/noisy, three unstable repeat cells. [W] |
| E15 | Overflow-resistant weighted means | Rejected: deep cancellation lost a representable `2e-151` mean, producing zero. [Q] |
| E16 | Normalize wholly subnormal compression mass | Rejected: independent tiny-value oracle failed. With minimum-subnormal `u`, values `[-3u,2u]`, weights `[u,u]`, the correct rounded mean is zero; candidate produced `-2u`. [Q] |
| E17 | Cumulative-mass quantile brackets | Rejected: values 0…5 with equal 0.1 weights changed correct median 2.5 to 2.0; CDF-gap roundtrip also regressed. [Q] |
| E18 | Symmetric compression direction | Rejected: rank/value/CDF regressions at matched size; sum metadata changed in seven of 72 states. [Q] |
| E19 | Gap-aware adjacent cap coalescing | Rejected: rank/CDF improved in 24 of 72 states, but value errors worsened in nine; one value-error maximum increased from 0.01436 to 0.44020. [Q] |
| E20 | Sweep sorted batch-CDF queries | Rejected: sorted 4,096-query case 4.80% slower. Later 57 cells, capacity witness and heldout work were skipped. [N] |

## Reconsideration priorities under the new server objective

These are candidates to validate, not accepted patches or a validated combination with PR #9. Keep the original decisions and measurements intact, then add a separately specified evaluation under the new policy.

| Priority | Why investigate | Known cost and unfinished work |
| --- | --- | --- |
| E03: two-way merge | Qualified longer-repeat delta300 cells used 40.81–44.05% less time; 21/24 primary repeat cells improved. Relevant to pairwise merge trees. | Five small f64 p50 controls cost about 0.39–0.49 ns/call by difference of arm medians. Fresh build/p50 completed; CDF/k-way/two-way/sequential confirmation matrices stopped. |
| E08: multiway merge | Qualified longer-repeat gains about 4.23–18.05%; 19/24 primary repeat cells improved. Relevant to aggregating many worker outputs. | Three small f32 batch-CDF controls cost about 43–65 ns/batch on roughly 1.2 µs baseline operations. Fresh build/p50/CDF completed; three merge matrices stopped. |
| E09: capped sequential merges | Seven `cap64Use` merge cases used 7.38–9.92% less time, around 90–170 µs saved per measured reset-inclusive operation. Trial allocations eliminated in 84 actual-hit diagnostic rows. | Two p50 costs around 0.39–0.42 ns/call; later CDF/codec/fresh confirmation incomplete. All 156 declared diagnostic rows passed exactness, branch parity and memory checks. |
| E13: Python serialization | Six qualified longer-repeat Python encode cells used 42.87–57.28% less time. Can rank higher if Python serialization dominates the real workload. | Existing patch also affects Rust encoding, adding roughly 16–43 ns in the ten regressing native cases. Native decode controls, explicit V1/V2 and edge-fixture codec supplement, and fresh Python heldout were stopped. Do not describe the existing patch as affecting Python alone. |

E02 remains a lower-priority simple optimization with narrower demonstrated gains. E10's few gains are concentrated in singleton/lifecycle cases; its broad throughput case is weaker. None of E03/E08/E09/E13 introduces or proves many-core scaling. The mechanism behind unchanged query-control slowdowns is unproven; do not label them harmless compiler/layout effects as a fact. Absolute cost differences are from arm medians, while reported percentage gates use paired ratios.

For a new evaluation, measure independent concurrent groups, one large dataset partitioned across workers, and ingestion→merge→query→serialization workloads. Compare pairwise trees and multiway reductions, including unequal shards and skew. Use increasing allocated worker counts; report physical cores, SMT, quotas and NUMA placement. Measure both fixed-total-data completion time and growing-data throughput, latency, merge contribution, peak RSS, memory per worker and allocation activity. Check accuracy/state/wire behavior across worker counts and merge orders. Investigate bandwidth, allocation/lock contention, Python GIL boundaries and nested thread-pool oversubscription where relevant. Extra RAM can be an acceptable trade-off; it is not evidence of unlimited bandwidth or memory budget.

## Modern-engine defects remain open

These are separate from the preserved original compatibility engine. Passing its original suite does not certify the modern engine on all data.

- `Use` capacities 1/2 can lose interior mass; E01's attempted repair worsened approximation errors.
- Positive residual bucket tails can be lost. A tail can contribute a meaningful first moment even when adding its weight rounds away; E06's attempted repair did not satisfy the quality gates.
- Extreme finite weighted products, centroid means and raw-sum representability remain problematic. E15's fallback introduced its own deep-cancellation regression.
- Wholly subnormal weights can cause underflow/coarse rounding of **intermediate weighted products**, corrupting ordinary-valued centroid means. E16 improved some fixtures but failed a separate tiny-value oracle. Baseline +0 versus exact −0 was numerically equal in that rejection check; candidate −2u was not.
- Some fractional-weight quantiles violate reflection symmetry. E17's equal-0.1-weight failure is a candidate-introduced regression, not the baseline defect.
- Tight-storage and repeated-merge approximation errors can be large. Compare rank error, value error and CDF error separately at matched settings and actual storage; do not hide scale-dependent differences behind one blanket epsilon.

Relaxed performance trade-offs do not authorize silent numerical or wire changes. No new accuracy algorithm from this campaign was retained, and there is no evidence supporting a “best in the world” claim.

## PR #9 / E12 validation scope

Saved published head: `42d620428520a74fec8c57d5dc6d4db0f6b1920e`, based on `25c8c16…`. The tested commit was `d54096087a9e975ab07faf71ab2db412a48edcfa`; the published head added a README clarification, with production/tests/CHANGELOG unchanged.

Recorded Linux checks: `make lint/build/test`; 121 Rust tests plus three pre-existing ignored tests, 352 Python coherence tests, 13 Java tests, 91 untouched upstream tests. Actual optimized ABI3 manylinux_2_28 wheel installation covered CPython 3.10–3.12, NumPy 1/2 and a Polars-free cell: 2,022 repeated runtime test instances, 36 ownership/value regression instances and 72 frozen output audits. This is not 2,022 unique test definitions.

Fresh qualified NumPy 2.3.4 cells showed CDF elapsed reductions of 44.86–54.76% and quantile reductions of 36.53–51.39%, six qualified cells each. Heldout 36-cell results: 25 improved, seven neutral/noisy, two unstable, two short. No certified regression; inconclusive encode/empty-CDF/weight-getter cases remain documented. These are single-worker WSL results, with no NumPy 1 speed claim, no Rust-core accuracy/speed gain, and no allocator-peak/RSS proof. The source archive/lock was checked; a separate wheel rebuilt from that source archive was not claimed for this PR.

Deployment still needs decisions: Python 3.13+, every platform wheel and minimum-glibc coverage are not proven here; shared distributions/import paths make independent co-installation unsafe; Rust root-path consumers need the documented nested package. Personal repository CI is not a substitute for upstream required checks or maintainer rollout approval.

## Evidence index and what to carry to the new machine

The following filenames are relative to the **Windows research root** above. Preserve referenced raw Linux artifacts, frozen protocols, patches, source snapshots and build/source proofs as well as these readable exports. Some ledger patch fields are null; that is an inventory gap, not proof that a prototype is absent. Root's fresh branch/patch inventory should resolve it before departure.

| Key | Readable evidence files |
| --- | --- |
| Overview | `day-experiments-state.json`; `day-experiments-completion.json`; `day-final-portfolio-audit-accuracy.json`; `server-performance-priorities.json` |
| N38 | `day-e03-primary-repeat-comparison.json`; `day-e08-primary-repeat-comparison.json`; `day-e03-e08-primary-repeat-review.json`; `day-native-declared-controls-summary.json`; `day-merge-shapes-validation-result.json`; `day-e03-e08-heldout-summary.json`; `day-e03-e08-heldout-final-review-speed-r3.json` |
| N2 | `day-native-remaining-v2-summary.json`; `day-e02-heldout-rejection-summary.json`; `day-supplement-untimed-result.json` |
| N | `day-native-remaining-v2-summary.json`; `day-e09-inventory-result.json`; `day-supplement-untimed-result.json` |
| NB / B | `day-native-followup-b-summary.json`; `day-breadth-summary.json` |
| L | `day-lifecycle-screen-summary.json`; `day-supplement-untimed-result.json` |
| W | `day-wheel-summary-v2.json`; `day-wheel-repeat-summary.json` |
| C | `day-wheel-repeat-summary.json`; `day-e13-native-rejection-summary-v2.json`; `day-e13-codec-supplement-review-freeze.json` (prepared supplement, not executed) |
| R | `day-e12-final-validation-result.json`; `day-e12-root-publication-check.json`; `day-e12-heldout-repeat-result.json`; `day-e12-acceptance-interpretation.json` |
| Q | `day-final-portfolio-audit-accuracy.json`; `day-portfolio-story-20261015T125731Z-v2.json` for the corrected defect inventory; raw paths below. The older story's candidate statuses are a superseded interim snapshot. |
| Continuation | `day-native-e12-interaction-plan.json` (unexecuted conditional plan); `day-survivor-validation-checklist.json`; `day-survivor-memory-storage-checklist.json`; `server-performance-priorities.json` |

Important raw paths relative to the **Linux research root** above:

- `patches/`, `validation/`, `harness/`, `evidence/v3/builds/`: preserve prototypes, focused regressions, frozen drivers and immutable build/source proof. Use original recorded hashes before reusing a patch.
- `evidence/repair-quality/comparison.json`; `evidence/quality-hypotheses/INDEX.json`; `validation/E15/RESULT.json`; `validation/E16/tiny-value-rejection/RESULT.json`; `validation/E17/v2-proposal-rejection/RESULT.json`; `evidence/v3/quality-screen-e18-e19-20261015/E18-quality.json`; `evidence/v3/quality-screen-e18-e19-20261015/E19-quality.json`.
- `evidence/v3/root-e03-e08-heldout-batch/INDEX.json` and `PLAN.json`: 1,440 measured cells, 2,016 explicitly skipped, eight certified regressions; zero reported state/quality/invariant/logical-storage mismatches in completed operations.
- `evidence/v3/root-e02-heldout-batch/`; `evidence/v3/E13-e13-v3-screen-encode-scale1/summary.json`; retain the surrounding raw records and source identities referenced by their corresponding exports.
- `evidence/supplement-v2/`; `evidence/e09-entry-allocation-v1/`; `evidence/merge-shapes-v1/`; `evidence/wheel-screen-v2/`; `evidence/e12-final-release/`.

Integrity anchors for the principal readable exports:

```text
day-experiments-completion.json                 5e2d15341c0d7f9323b4210ec92e47efae62067de44c6acd1c62c778ab39e6a0
day-e03-e08-heldout-summary.json                 5146c3f70dc4ce39cf58190a658a251797756b41cdfd4f1ec90b6fd9bf5badf5
day-e03-e08-heldout-final-review-speed-r3.json    8cedd513d2f34795263f892cb321029ccf72f81ffd4f98dcb11944a7d9da9bae
day-e02-heldout-rejection-summary.json           a977934c9379dd0bfc3139259c0a84d8e24a7ed568d0a04dda2bdbe6e5d36da2
day-e12-final-validation-result.json             97bf2d2459b7155e80c50c045c92b66205adfef65ff90e1d1e85f26aa546ed85
```

## Next steps after transfer

1. Verify remote branch/PR identities and the backup manifest; check out the intended review branch in an isolated Linux clone. Keep historical patches/results and failed harness attempts intact. Review `AGENTS.md`, design docs and CHANGELOG requirements before edits.
2. Review the compatibility stack and PR #9 in order. Check the live bases before restacking; do not merge or publish a release as part of evidence transfer.
3. Define a new, explicit production-workload acceptance policy with absolute costs, throughput objectives and memory reporting. Preserve the old zero-regression conclusions as historical evidence.
4. Revisit E03/E08, then E09, with E13 conditional on Python serialization cost. Complete the missing controls/heldout coverage and validate combinations with E12 before recommending integration. Existing patches were measured individually against the old baseline, not as a combined server-ready library.
5. Run release, compatibility, cross-language and actual installed-wheel gates on the exact resulting head. Measure scaling only on the hardware actually available; distinguish preparation from demonstrated >100-CPU behavior.

This guide was drafted from saved evidence and peer-reviewed. Migration-time branch and archive verification is recorded separately in HANDOFF.md and the migration receipts.
