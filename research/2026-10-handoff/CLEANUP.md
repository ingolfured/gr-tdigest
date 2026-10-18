# Local cleanup and recovered historical work — 18 October 2026

After the initial handoff, the user authorized preserving useful changes and cleaning all dirty tdigest checkouts, explicitly including the formerly protected original checkout. This supersedes the earlier instruction to leave that checkout dirty. Current product PRs remain the review path; nothing from this historical recovery is approved for integration.

## What had value

- The original checkout had 16 modified tracked files and 18 nonempty untracked source/document files. All 34 are preserved exactly in a separate archive commit. Eight untracked compatibility files match PR #9 byte for byte. The tracked changes are earlier delta/adapter work carried forward or superseded; the audit found no missing current production fix or regression test.
- Seven old documents are useful historical drafts but include stale median, f32-weight, atomic-provenance and 70-pass/21-fail compatibility claims. Follow the current compatibility guide instead. The old top-level pure-Python `tdigest_rs` shim ignores delta and substitutes a trimmed mean after merging; never reactivate it as the compatibility implementation.
- Three previously unpublished stashes are now preserved as exact Git merge commits, including all parents. The first contains an automatic scale-selection prototype and quality tests in its third parent. These two files are not in the top commit tree. The other stashes preserve Python binding refactoring and shared-centroid/Polars decode experiments. All are unvalidated historical research; the old decode-cache experiment has a potential metadata-reuse defect.
- Three old local branch heads now have archive refs in this repository. The old PR38 head was already archived in the upstream fork, and is also preserved here to make recovery from the original clone explicit.
- A unique generated Python lockfile from `/home/ingo/git/tdigest-rs/bindings/python/uv.lock` is preserved as [upstream-python-uv.lock](cleanup/upstream-python-uv.lock), SHA256 `3cdf3703e3d8605c819a711206d86bc1e182b9268c3e290d22f7084e71f80082`. It belongs to historical base `52cda0f4341df6a2befbd32639f90846b18d60da`; it is not the current project lock.

## Recovery refs

All refs below are in `ingolfured/gr-tdigest`. A normal clone or `git fetch origin` retrieves them. These are archival refs, not proposed merges.

| Branch | Commit | Content |
| --- | --- | --- |
| `archive/20261018/original-dirty-checkout` | `ea0b0dd46c5e39bca69c915c8be9cf1107852cf0` | original_worktree |
| `archive/20261018/original-stash-0-wip-before-v0.2.4-release` | `3adf4e8a4b629ad0d4b7cfa50b984ad78c3e68e3` | stash |
| `archive/20261018/original-stash-1-new-world-837a7c8` | `6a6fb4f31b84ede48815b15c302ab3a47c3e29a6` | stash |
| `archive/20261018/original-stash-2-new-world-c9f8a6d` | `59a782557e8482a7667f7a3eb25bad41261dba5d` | stash |
| `archive/20261018/original-local-pr38-docs-compat` | `e8562fd1c25a33cfd09359f671da126837c7bba4` | historical_branch |
| `archive/20261018/original-local-bigbang-review-1-core-delta` | `90962e90a09fc7fe72f43290ff17db808930560b` | historical_branch |
| `archive/20261018/original-local-bigbang-review-2-python-api` | `1c043b16b19f526a3213e1bf585ba14a4fe11f61` | historical_branch |

To recover the first stash including its index and two untracked source files, use a new isolated worktree at its original base:

```bash
git fetch origin
git worktree add --detach ../tdigest-autoscale-recovery 9becca2e6000fa2097e1ad0ab46a5019dbed7e8e
git -C ../tdigest-autoscale-recovery stash apply --index origin/archive/20261018/original-stash-0-wip-before-v0.2.4-release
```

This intentionally recreates that old work in progress, not current production code. To read only the two extra files, inspect `origin/archive/20261018/original-stash-0-wip-before-v0.2.4-release^3:src/tdigest/auto_scale.rs` and the matching `src/quality/auto_scale_quality.rs`. The [stash audit](cleanup/cleanup-stash-audit.json) records all three bases, parents, paths, and limitations.

## Cleanup state

- All 20 prototype worktrees and the harness are clean. Each existing `experiment/day20-*` local branch now points at its already-pushed archive snapshot; source bytes were checked before changing the branch/index. The harness additionally uses the archive's synthetic CHANGELOG entry. No new experiment was accepted by this operation.
- `/home/ingo/git/gr-tdigest` is clean on its original `master` commit `8dea4cf5e336db03eeefc18cad4b5f8537822dd2`. Source restoration/removal followed remote publication and independent fetch/byte verification. Removed `.tdigestrs-target`, `.gradle-tdigestrs`, and the empty `.codex` marker. Its three stashes were cleared only after their full parent topology and extra source objects were fetched back from GitHub and verified.
- Four already-missing `/tmp/gr-tdigest-review-prs/*` worktree registrations were pruned; their branch commits remain preserved.
- The additional upstream checkout cleanup follows publication of its lockfile; the final cleanup receipt records completion and the nested-clone destination.
- Ignored local settings, environments, existing build reports, and the explicitly local-only `typical_usecase.md` remain local. No blanket ignored-file deletion or upload was performed. Git-clean does not mean every generated file was removed from disk.

## Evidence and scope

The [original audit](cleanup/cleanup-original-review.json), [experiment parity audit](cleanup/cleanup-worktree-audit.json), [source cleanup receipt](cleanup/cleanup-applied.json), [stash preservation receipt](cleanup/cleanup-stashes-preserved.json), and [ref map](cleanup/preserved-refs.json) provide exact identities. The original archives and their checksums have not been rewritten; this supplement is committed directly alongside them. Historical records saying the original checkout was untouched describe the initial migration, before this later user-authorized cleanup.

This is preservation and filesystem cleanup, not a new product release or numerical validation. No builds, tests, or benchmarks were rerun. PR #6/#7/#8/#9 and upstream #38 heads and bases are preserved.
