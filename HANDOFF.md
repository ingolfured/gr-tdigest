# Start here: t-digest handoff

This branch preserves the project for continuation on another computer. It is based on the validated personal PR #9 and adds documentation and historical research only. **Do not merge this research archive wholesale into the product or upstream PR.**

## Resume from a fresh Linux clone

```bash
git clone --branch research/handoff-20261018 https://github.com/ingolfured/gr-tdigest.git gr-tdigest-handoff
cd gr-tdigest-handoff
python3 research/2026-10-handoff/verify_archive.py --check-refs --extract ../tdigest-research
```

The archive verifier requires Python 3.9 or newer (independent of the library's tested Python 3.10–3.12 matrix). It checks every archive part and extracted file against SHA256 records, checks the personal branch mappings against fetched refs, and restores executable modes. It never executes historical scripts or binaries. The extraction destination must be new. The archive contains raw results and frozen executables as well as documents; see [sizes and scope](research/2026-10-handoff/archive-summary.json) before extracting. A normal clone fetches the archive refs; do not use `--single-branch` with `--check-refs`.

Read this file, [experiment results and priorities](research/2026-10-handoff/EXPERIMENTS.md), [compatibility and release evidence](research/2026-10-handoff/COMPATIBILITY.md), and the [latest server-performance preference](research/2026-10-handoff/reports/server-performance-priorities.json). Then read `AGENTS.md`, `api_design.md`, `tdigest_design.md` and the current [review guide](compat/REVIEW_GUIDE.md).

For subsequent product work, create a separate worktree from the code branch:

```bash
git worktree add -b continue/server-throughput ../gr-tdigest-work origin/review/4-numpy-output-copy
```

## Branch and PR structure, checked 18 October 2026

| Order | Personal PR | Branch | Head | Base branch |
| --- | --- | --- | --- | --- |
| Foundation | #1 and #2 already merged | `master` | `cd4f14c4924ec31264db3a3e8243011b204d268a` | — |
| 1 | [#6](https://github.com/ingolfured/gr-tdigest/pull/6) | `review/1-digest-correctness` | `6eb4203fe1c4ad8d337636d5aca549e0fec155b3` | `master` |
| 2 | [#7](https://github.com/ingolfured/gr-tdigest/pull/7) | `review/2-legacy-python-api` | `911dc35493f1774fed1320f9da30d553b41f59b2` | `review/1-digest-correctness` |
| 3 | [#8](https://github.com/ingolfured/gr-tdigest/pull/8) | `review/3-upstream-compatibility-gate` | `25c8c1642029a640b153c160c22a89795d29ad91` | `review/2-legacy-python-api` |
| 4 | [#9](https://github.com/ingolfured/gr-tdigest/pull/9) | `review/4-numpy-output-copy` | `42d620428520a74fec8c57d5dc6d4db0f6b1920e` | `review/3-upstream-compatibility-gate` |

All four are open. Personal #3, #4 and #5 are closed/superseded. Review #6 → #7 → #8 → #9, preserving stacked bases; retarget/restack dependents only after checking the actual merge strategy and current remotes. Do not infer that an old `/3` title means #9 is part of that original three-PR split. Full PR titles, descriptions and current recorded refs are in [pull-requests.json](research/2026-10-handoff/pull-requests.json).

The upstream submission is a **different repository and adapted tree**:

- [G-Research/tdigest-rs #38](https://github.com/G-Research/tdigest-rs/pull/38) is open and draft.
- Its head is `ingolfured/tdigest-rs:overwrite_snapshot_v1_0_0`, at `57e7978d0811baed4a546ff2f08dccaeff03c321`; base is G-Research `main`.
- Clone the fork separately with `git clone --branch overwrite_snapshot_v1_0_0 https://github.com/ingolfured/tdigest-rs.git tdigest-rs-upstream38`.
- Personal PR #9 has not been ported into #38. Preserve its package adaptation and source-specific validation when updating it; do not force one repository's tree over the other.
- The independent original oracle remains pinned at `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`, also reachable from the fork's `main` at this snapshot. Keep its original source/lock and expected outputs independent.

## Experiments are preserved, not all accepted

The 15 October campaign implemented 20 prototypes. Under the original strict per-case rule, **E12 was retained as PR #9; 19 were rejected**. Later validation was skipped after decisive failures. Historical results are unchanged.

On 16 October the user relaxed that performance rule and emphasized production on servers with **more than 100 CPUs and substantial RAM**. A local slowdown or extra memory can be acceptable when representative workload throughput benefits. This does not make the old rejected prototypes validated winners and does not relax original API/numerical compatibility.

Next priorities are E03 (two-way merge), E08 (multiway heap merge), E09 (cap search allocations), and conditionally E13 (Python serialization). **E09 is an experiment, not PR #9.** None proves many-core scaling. Complete missing validation and measure interactions with E12 before integration. Keep known modern-engine numerical defects visible; the old API's passing suite does not resolve them.

Each dirty experiment worktree was preserved as an `archive/20261018/day20-*` branch, including untracked tests and the harness. These commits are archival snapshots, not validated production commits. Existing working files, indexes and branch heads were left alone. Clean local-only historical commits were also preserved or mapped to an already-pushed equivalent. The complete [branch map](research/2026-10-handoff/branch-map.json) gives old names, original heads, new remote refs and snapshot commits; the [experiment map](research/2026-10-handoff/EXPERIMENT-BRANCHES.md) gives one route for each E01–E20.

For example, inspect E03 independently:

```bash
git worktree add --detach ../tdigest-e03 origin/archive/20261018/day20-two-run
```

## Research layout and historical paths

`research/2026-10-handoff/` contains the two human-readable guides, current PR/branch records, selected readable JSON reports, and the complete checksummed research archive. Extraction creates:

| Extracted prefix | Original source |
| --- | --- |
| `work/` | Windows research `C:/Users/ingol/Documents/Codex/2026-10-14/hi/work`, also accessed as `/mnt/c/Users/ingol/Documents/Codex/2026-10-14/hi/work` |
| `day/` | `/var/tmp/codex-tdigest-pr3-20261014/day20261015` research, patches, validation, frozen harnesses and final artifacts |
| `overnight/` | `/var/tmp/codex-tdigest-pr3-20261014/overnight` compatibility, oracle, accuracy and release evidence |
| `logs/` | `/var/tmp/codex-tdigest-pr3-20261014/logs` |
| `numerical-parity/` | `/var/tmp/codex-tdigest-pr3-20261014/numerical-parity` source/lock/runner |

The Git checkouts formerly at `.../work/gr-tdigest` and `.../work/tdigest-rs-upstream38` are reconstructed from the two repositories, not extracted over an existing checkout. The day prototype source checkouts are reconstructed from archive refs; their loose patches and frozen build-source records are also archived. See [relocation.json](research/2026-10-handoff/relocation.json).

Historical JSON and scripts intentionally retain their original absolute paths and checksums. **Do not mass-rewrite the evidence or blindly execute the old one-shot `work/*.py` helpers.** Read the relevant runner, provide a new adapter/configuration for destination paths, and record new runs separately. The archive retains failed attempts, negative controls and superseded drafts. In particular, extracted `work/REVIEW_GUIDE.md` is stale; use the current committed `compat/REVIEW_GUIDE.md` and these handoff guides.

The archive excludes dependency environments, ordinary build caches, Git control directories and bytecode; frozen executables outside those caches are included because their exact hashes are part of the evidence. Full excluded-path reasons are in `archives/MANIFEST.json.gz`. Toolchains and dependencies must be installed anew. Archive parts are normal Git files, not LFS pointers or temporary download links.

## Validation and continuation boundaries

- This migration does not change product source, tests, numerical fixtures, manifests or locks relative to PR #9. Migration verification covers archive contents, snapshot source parity, branch mapping and clean-clone restoration; prior product validation is described with exact source/artifact identities in the guides.
- All library builds/tests/benchmarks must run on Linux/WSL. Read the Makefile and use `make setup`, `make build`, `make test` and `make lint` as appropriate for new code; actual installed release-wheel checks remain required before claiming a new release candidate. `make lint` can modify files.
- Existing `tdigest_rs.TDigest` and `tdigest_core` use pinned original arithmetic. Modern `gr_tdigest` remains opt-in. Keep the untouched upstream suite and independent oracle. No blanket epsilon across data scales.
- Measure both concurrent independent digests and sharded datasets including merge costs, at increasing available worker counts. Report topology/NUMA/SMT, throughput, latency, memory, and separate rank/value/CDF errors. No 100+ CPU result exists yet.
- The original dirty checkout `/home/ingo/git/gr-tdigest` was not touched or included in this migration. Its unrelated personal changes are outside this research handoff.
- The old overnight/day schedules are complete. No background benchmark or automation is transferred or restarted. Do not merge PRs or publish a release without the user's instruction.

Suggested first prompt for the next agent:

> Read HANDOFF.md on research/handoff-20261018. Verify the archive and current remotes, then use a separate Linux worktree from review/4-numpy-output-copy. Preserve the original tdigest_rs API and oracle. Continue the E03/E08/E09 evaluation under the user's many-core, workload-based performance objective, keeping historical regressions and unfinished gates explicit. Do not merge or release.
