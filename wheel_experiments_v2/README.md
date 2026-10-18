# Installed-wheel experiments

Prepared code-only. Coordinate CPU ownership before invoking any command, including imports and audits. All commands use WSL ArchFresh as ingo.

```
DAY=/var/tmp/codex-tdigest-pr3-20261014/day20261015
RUN=$DAY/harness/wheel_experiments_v2/run.py
python "$RUN" --baseline "$DAY/evidence/numpy-wheels/baseline/BUILD.json" --candidate "$DAY/evidence/numpy-wheels/e11/BUILD.json" --label e11 --experiment E11 --numpy np2 --limit 1 --audit-only
python "$RUN" --baseline "$DAY/evidence/numpy-wheels/baseline/BUILD.json" --candidate "$DAY/evidence/numpy-wheels/e11/BUILD.json" --label e11 --experiment E11 --numpy np2
```

Repeat with `--phase confirm` after exact promising screening. `--numpy np1` always audits and cannot produce a speed claim. `--suite target` and `--suite control` split work, but acceptance requires both. E12/E13/E14 reuse the same driver with their registered candidate manifests and experiment identifiers. Every result is write-once under evidence/wheel-screen-v2.

BUILD.json must contain the common `build_identity` before running. Preparation does not install packages, build wheels, import a candidate, or execute tests. Full parent-managed compatibility gates remain mandatory.

V2 adds scalar-result exactness after a v1 harness TypeError. All v1 sources and failure records remain unchanged.
