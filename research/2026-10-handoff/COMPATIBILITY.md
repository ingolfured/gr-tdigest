# t-digest migration handoff: compatibility and evidence

Prepared on 18 October 2026 from local evidence. Remote identities below are the **15 October recorded snapshot**, not a new remote check; use the migration inventory's fresh remote verification before continuing. This guide records completed work and the user's later server-performance priorities. It does not authorize merging or releasing.

## What must remain compatible

Existing applications keep `from tdigest_rs import TDigest`, the original delta arguments, dtype/integer-weight rules, merge/update and copy semantics, errors, and saved-state reconstruction. This path deliberately preserves the original numerical engine. The separate modern engine is explicitly selected through `gr_tdigest`. Modern speed improvements do not automatically accelerate old imports or make both algorithms return the same numbers.

The independent original source is pinned at **`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`**. Preserve its 35-file export, committed lock, provenance/hash guards, untouched **91-test upstream suite**, and independently generated golden data. Debug and release oracle records each cover **70 cases, 1,762 exact-bit comparisons and 126 NaN classifications**. Do not regenerate expected answers with a modified implementation or introduce one epsilon across all scales.

Original saved-state quirks remain part of compatibility. The genuine published **0.1.0** pickle fixtures are distinct from the pinned **0.2.0** source. The historical loader reconstructs using default delta and can recompress; compare against the old loader's restored result. NumPy 2 protocol-5 pickle rollback into NumPy 1 is a documented limitation.

The original Rust public package/API is also preserved through `tdigest_core`; an old repository-root path dependency still needs explicit package/nested-path selection. Inspect the destination Cargo manifests rather than assuming Python compatibility establishes Rust path compatibility.

## Review stack and source identities

The personal repository and upstream submission are separate trees. Keep their validation claims tied to their own commits and artifacts.

| Review | Purpose | Recorded head/base |
| --- | --- | --- |
| [Personal #6](https://github.com/ingolfured/gr-tdigest/pull/6) | Modern weighted mass, exact point means and state corrections | `6eb4203fe1c4ad8d337636d5aca549e0fec155b3`, based on `master` |
| [Personal #7](https://github.com/ingolfured/gr-tdigest/pull/7) | Original Python API and arithmetic | `911dc35493f1774fed1320f9da30d553b41f59b2`, based on #6 |
| [Personal #8](https://github.com/ingolfured/gr-tdigest/pull/8) | Original Rust API, runtime/artifact gates and measured evidence | `25c8c1642029a640b153c160c22a89795d29ad91`, based on #7 |
| [Personal #9](https://github.com/ingolfured/gr-tdigest/pull/9) | E12: NumPy output conversion | `42d620428520a74fec8c57d5dc6d4db0f6b1920e`, based on #8 |
| [G-Research #38](https://github.com/G-Research/tdigest-rs/pull/38) | Actual upstream package adaptation | `57e7978d0811baed4a546ff2f08dccaeff03c321`, branch `overwrite_snapshot_v1_0_0` in `ingolfured/tdigest-rs`; recorded open/draft |

Read **#6 → #7 → #8 → #9**. Preserve stacked bases when reviewing/restacking. #9 had not been incorporated into upstream #38. The recorded #9 CI run `38737554737` succeeded; the upstream fork's run `38693154198` succeeded. Neither substitutes for current upstream-required checks or approval.

## Validation that already exists

| Source/artifact | Completed evidence | Important boundary |
| --- | --- | --- |
| Personal E12 tested commit `d54096087a9e975ab07faf71ab2db412a48edcfa` | Linux `make lint`, `make build`, `make test`; 121 Rust tests, 352 Python/coherence tests, 13 Java tests and all 91 untouched upstream tests | Three pre-existing Rust tests remain ignored. Published #9 differs only by a README clarification; production/test hashes are unchanged. |
| E12 optimized release wheel | 2,022 installed test instances across seven isolated runtime environments; 36 additional E12 regression instances and 72 frozen-baseline exact audits | Repeated environments are not distinct test definitions. CPython 3.10–3.12, NumPy 1/2, Polars and a separate no-Polars cell were tested. |
| Actual upstream package source `daad31585d8d88cbffeeeba206ccfeebd733e9a1` and later documented successors | 336 Python/coherence + 91 original tests, 121 Rust and Java; direct wheel **and separately source-rebuilt wheel each** pass 2,036 installed instances | This is upstream-distribution evidence, not the personal E12 wheel. Later LF/test-launcher changes and their checks are separately recorded. |

E12 wheel SHA256: `dab5447ed271cf99478d4546b00ab21c7c5dfc5e2c1c1c92aa8c1d66d37742b0`. Its filename is `gr_tdigest-0.2.4-cp310-abi3-manylinux_2_28_x86_64.whl`. Its genuine source archive was checked against committed source/lock; **the E12 gate did not separately rebuild a wheel from that archive**. Do not transfer the earlier upstream rebuild claim to it.

E12 bulk-copies existing Float64 results into NumPy-owned arrays without changing arithmetic. In qualified installed NumPy 2.3.4, single-worker Linux cells, batch CDF elapsed time fell **44.86–54.76%** and quantile elapsed time **36.53–51.39%**, relative to the pinned modern baseline. There were six qualified cells per operation. These are time reductions, not universal throughput or accuracy claims.

The 36-cell fresh repeated E12 matrix had 25 improvements, seven neutral/noisy cells, two unstable cells and two short cells, with no certified regression. Preserve the inconclusive f64 encode control, empty-f32 CDF and two large weight getters. NumPy 1 was compatibility-tested, not performance-qualified. Equal array payload/ownership was checked; allocator usable bytes and peak RSS were not measured. Scalar behavior, including one-element NumPy queries, remains unchanged.

## Completed experiments and newer direction

The 15 October campaign screened **20 implemented prototypes**: 19 rejected, only E12 retained. The other prototypes were not integrated; later tests were deliberately skipped after decisive failures. In particular, fresh controls rejected E02 (+9.71% median time in one p50 cell), E03 (five p50 regressions, +3.75–6.71%) and E08 (three batch-CDF regressions, +4.37–5.02%) under the then-fixed per-cell rule. Preserve all target gains, regressions, uncertainty, source hashes and rejected patches.

**The user changed the performance objective on 16 October.** Large servers with more than 100 CPUs and substantial RAM are the intended target. Measured local slowdowns or extra memory can be acceptable when representative end-to-end throughput benefits. This is a new workload-based evaluation, not permission to relabel historical rejections. The original API/arithmetic, independent oracle and correctness requirements remain protected.

Read `work/server-performance-priorities.json` before planning further experiments. Many-core scaling has **not** been measured. E03/E08 are merge changes, not parallel execution implementations. New work should characterize actual physical/SMT cores, CPU quotas and NUMA topology; cover independent groups and sharded-dataset aggregation; measure throughput, latency, memory per worker/peak memory, serialization, merge costs and accuracy across worker counts. Control nested thread pools and preserve caller-managed concurrency. Do not rerun old schedules automatically or assume their single-worker conclusions establish server behavior.

## Unresolved issues and release decisions

The following concern the **modern engine**, not failures demonstrated in the frozen original path:

- Tiny `Use` capacities 1/2 can lose interior mass. E01 restored conservation but worsened rank/CDF or individual quantile accuracy, so its repair was rejected.
- Bucketization can omit a positive residual tail. E06 improved the weighted moment but worsened value/rank/CDF errors at the same storage. A sub-ULP tail and an unrepresentable aggregate mass need separate accounting.
- Finite inputs can overflow weighted products/centroid means; raw-sum representability is a separate limit. E15's fallback lost a deep-cancellation residual.
- Wholly subnormal weights can underflow or coarsely round **intermediate products**, corrupting ordinary-valued centroid means. E16's candidate-only tiny-value failure must not be mislabeled as a baseline failure.
- Fractional quantile accumulation can violate reflection symmetry. E17 improved selected fixtures but introduced another median error; its proposed repair was rejected.
- Tight-storage separated-mode and repeated-merge accuracy remains limited. Rank/CDF and value error can move in opposite directions; this is not automatically an API-contract defect.

Modern f32 means already use f64 weights: the measured centroid size increased from 12 to 16 bytes; f64 centroids remained 24 bytes on the tested target. Modern Polars and V3 extension rules preserve this mass. Old readers cannot silently accept every extended state. A same-delta comparison is not necessarily matched storage or matched error.

At the recorded upstream closeout, a maintainer still needed to re-enable the inactive upstream workflow, run current-head checks and supply required approval. Version `2.0.0`, publication order, package ownership and deployment support remained release decisions. Recheck their current state; do not infer readiness from fork CI.

`gr-tdigest` and `tdigest-rs` share import paths and must not be independently co-installed. Use a fresh environment and one distribution owner. Python 3.13+ was unverified/excluded; builds of other platform wheels do not prove their installation or minimum-glibc deployment. See source-specific rollout evidence rather than broadly declaring platform support.

## Minimum evidence to carry to the new computer

Use a hash manifest and preserve relative layout. Absolute `C:/...` and `/var/tmp/...` paths in receipts are historical provenance; a copied JSON pointer alone does not transfer its referenced artifact.

| Keep | Purpose |
| --- | --- |
| `work/overnight-state.json`, `work/day-experiments-state.json`, `work/server-performance-priorities.json` | Compatibility campaign, complete experimental dispositions and latest user priorities. |
| `work/day-experiments-completion.json`, `work/day-final-portfolio-audit-accuracy.json` and its remote addendum | Compact final story and independently checked limitations/evidence references. |
| `work/day-e12-final-validation-result.json`, `work/day-e12-documentation-amendment.json`, `work/day-e12-root-publication-check.json`, `work/day-e12-acceptance-interpretation.json` | Tested/published source identity, installed artifact results and honest performance scope. |
| `work/day-final-remote-verification.json`, `work/day-final-branch-verification.json`, plus migration-time replacements | Historical PR/branch identities; current remote inventory must remain distinguishable. |
| `work/day-e02-heldout-rejection-summary.json`, `work/day-e03-e08-heldout-summary.json`, `work/day-e13-native-rejection-summary-v2.json`, and the completion report's other evidence references | Preserve decisive failures and skipped work; do not carry only successful measurements. |
| Repository `src/upstream/PROVENANCE.md`, untouched suite, original Rust/API tests, golden data, relevant `compat/` review/validation reports | Reproducible compatibility contract, preferably from pinned Git commits rather than loose draft copies. |
| Independent oracle source/lock/runner/golden/manifest/results and debug/release outputs; genuine published pickle fixtures and provenance | Preserve independent expected results. Historical oracle directory: `/var/tmp/codex-tdigest-pr3-20261014/overnight/numerical-review/original-committed-lock-oracle/`. |
| Source-bound raw release/support results, wheel/sdist hashes and artifacts if exact artifact reuse is intended; frozen benchmark protocols/source manifests/results and rejected patches for proposed reevaluation | Summaries establish history; raw evidence and exact source are needed for detailed review or repeatability. Large build caches/venvs are not substitutes. |

Useful Windows fixture copies include `work/upstream_numerical_golden.json` and `work/overnight-compat/published_pickle_bundle/data/published_0_1_0_pickle_golden.json`; verify against committed/frozen manifests before reuse. Do not blindly execute `work/*.py` helpers: many are historical one-shot mutation scripts with old absolute paths. Read and adapt the relevant script in a new location first.

## Resume safely

The original checkout **`/home/ingo/git/gr-tdigest`** was protected and dirty during the experiments. After the initial migration the user authorized cleanup; its source and stashes are now remotely archived and it is clean. The [cleanup supplement](CLEANUP.md) supersedes the old no-cleanup instruction and records exact recovery refs. Prior work used `/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest` and isolated siblings, in WSL `ArchFresh` as user `ingo`. On the destination use new isolated checkouts, verify branch/remotes/HEAD/status and read repository instructions before edits. Keep library builds, tests and benchmarks on Linux; never silently reuse a stale shared Cargo target for different candidate sources. Coordinate heavy work so benchmark timings are uncontaminated.

Start from the current remote inventory and latest priorities, not an old automation prompt. The overnight and day experiment runs were closed. Do not merge PRs, publish releases, regenerate oracle fixtures or change original arithmetic as a migration side effect.

**Stale draft warning:** `work/REVIEW_GUIDE.md` predates the completed f64-weight, Python-support and upstream-refresh work. Some `work/overnight-compat/*/files/` documents are also preparatory drafts. Prefer documents committed at the verified PR heads and the source-labelled final receipts. Do not promote an old draft to current truth because its filename looks authoritative.
