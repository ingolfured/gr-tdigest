# Full-metric reducer: preparation, not execution

`reduce_heldout.py` is a new reporting-only reducer for the exact completed ff73cfa study. It invokes no native executable, retunes nothing and writes outside the immutable study. It requires independent numerical review and root's explicit CPU allocation before running.

Future command:

```text
python3 reduce_heldout.py --execute --cpu-slot-released \
  --study /var/tmp/codex-tdigest-pr3-20261014/overnight/storage-budget-study-ff73cfa \
  --metadata-report .../held-out-metadata-ff73cfa/policy-summary.json \
  --output FRESH_REPORT_DIRECTORY
```

The metadata report is pinned to SHA256 `14a827d8718f65cdc50985903ec0daeb0ac35b5690daf75ea30499e5deee1c9f`. Reviewed helper hashes and the training selection hash are required. Binary, resolved Cargo lock and every recorded source file are rehashed before/after reduction. The complete compressed filename set must match metadata. Every gzip file is decompressed once through a hashing reader that verifies all compressed bytes, and every raw state must match its case, policy evaluation, route and input fingerprint. The reducer compares all 1003 paired effective quantile queries and Type 7 references before calculating changes.

Budgets are 600 seconds of processing, 8 GiB decompressed bytes, 8 MiB per line, exactly 8064 expected records, and 64 MiB generated metric/summary output plus a small status manifest. Runtime is checked during bounded reads, per record, and each output write. Partial files survive a failed check or exhausted budget, and `status.json` is authoritative: consumers must require `status=complete` and verify its output hashes before interpreting a report as complete. A shell timeout may additionally bound unexpected interpreter stalls. The reducer has no resume or automatic retry policy.

It retains one case's raw query arrays at a time and at most 8064 compact state summaries. Outputs are `state-metrics.jsonl`, `route-pairs.jsonl`, `summary.json`, `raw-file-sha256.json`, and `status.json`. Group summaries preserve distribution, size, precision, policy, core and route, with the same three fresh seeds. Value errors are grouped by distribution, not pooled across unrelated scales. All 24 choices remain visible, including eight unavailable cells. The 2880 matching-route pairs are distinguished from 2304 k-way-versus-sequential comparisons.

Each compact state retains p50 estimate/reference/rank/value error, six predeclared tail points with rounded queries, worst queried rank/value points, modern-only worst CDF probe, logical bytes, capacity measurement semantics, wire size and raw filename/hash/record identity. Original out-of-range and decreasing-query observations remain visible. Missing/nonfinite results make aggregate maxima null with explicit counts and finite-only diagnostics; they never become zero or silently disappear. Query counts are descriptive, not independent trials. Logical bytes are not RSS or peak/retained allocation; original capacity in this accuracy harness is measured after clone-and-shrink, while modern capacity is unknown.

`test_reducer.py` contains synthetic no-core checks for equality/miss semantics, retained nonfinite results, query rounding, value arithmetic, single-pass compressed hashing, and output-budget preservation. No real samples or study results are required for those checks.
