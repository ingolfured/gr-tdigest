# Day experiments v3

This directory is separate from frozen v2. No build or timing has been run by the preparation script.
All commands below run in WSL ArchFresh as ingo, under the runner's exclusive CPU lock. Coordinate the CPU owner with the root agent first.

```
DAY=/var/tmp/codex-tdigest-pr3-20261014/day20261015
RUN=$DAY/harness/day_experiments_v3/run.py
python "$RUN" build --repo "$DAY/baseline" --label baseline-v3
python "$RUN" smoke --label baseline-v3
python "$RUN" build --repo "$DAY/cdf" --label e04-v3
python "$RUN" smoke --label e04-v3
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation cdf --limit 1
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation cdf_atomic
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation cdf_weighted
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation cdf
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation build
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation p50
python "$RUN" compare --label e04-v3 --experiment E04 --kind speed --operation cdf_batch
```

Repeat mandatory targets and controls with `--phase confirm` only after a promising exact screen. `--audit-only` records full correctness comparisons without timing; `--limit` cannot support retention. Matching short initial batches allow one explicit `--rounds-scale 10` run; both results remain evidence. This rule never permits changing the candidate or comparing a v2 binary with v3.

E10 targets are `frontend_owned_1`, `frontend_owned_2`, `frontend_owned_40`. Optional `frontend_clone_*` measures caller-reset-inclusive cost and is separately labeled. E20 uses `batch_ORDER_COUNT`, with the full preregistered matrix in protocol.json. Every result is under evidence/v3 and cannot overwrite another run.

Timing is single-thread native Rust, pinned to CPU0; interpretation is limited to that environment. Actual installed Python wheels need a separate protocol.
