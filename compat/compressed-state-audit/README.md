# Compressed-state invariant audit: historical failure and fix

The historical report retains the original **failing evidence**. The new fixed-source
report passes the same bounded invariant matrix; neither is an accuracy benchmark. The
recorded release wheel at source `22867bd55739a96a6fab3c8c48b27406ecb32197`
completed 1,440 routes and 3,840 checked states. Two states violated the strict
centroid-order invariant: f64, adjacent signed means, weights around `1e100`,
capacity 20, batch and reversed batch. They contain two equal centroid means
after compression. All other checked invariants passed in this bounded matrix.

The recorded failure is preserved as evidence of the defect in that source.
Commit `1d9993f` bounds a finite compressed average to its cluster's input interval,
preventing rounding from crossing into the next cluster. The final fixed-source
rerun is **PASS: 1,440 routes and 3,840 states, with zero failures** at `c5dec35`, with artifact
provenance at **[exact artifact validation](../validation/c5dec35-validation.json)**. The [new raw result](../validation/c5dec35-modern-compressed-state-audit.json) records the passing rerun. The audit is separate from the 800-digest
uncompressed weighted-point reference audit.

## Reproduce

With NumPy and the chosen `gr_tdigest` wheel installed in the active Python:

```sh
python audit_compressed_invariants.py --output result.json
```

For the passing rerun, install the wheel built from `c5dec35` and add
`--source-dir /path/to/gr-tdigest --source-ref c5dec35`. To reproduce the historical
failure, use the `22867bd` wheel and `--source-ref 22867bd` instead. Source flags
record provenance; they do not select or install a different native library.
The result records script/native hashes and optional source hashes. The source checkout was the candidate branch at `fae92a4` when recorded;
its five checked core source files matched `22867bd` exactly. The native hash
matches the preserved optimized `22867bd` release wheel.

The default deterministic seed is `202610140712`. Both mean precisions run with
capacities 10, 20, 40, 100, and 1000; duplicate-heavy, adjacent-positive,
signed-adjacent-with-duplicates, and varied-spacing values; and integer, binary
and decimal fractional, large integer, small normal, and large normal weights.
Each route uses 40 weighted points. Routes are batch, reversed batch,
incremental weighted add, 40-way merge, shuffled 40-way merge, and incremental
merge. Incremental paths also check states after 1, 2, 3, 10, and 20 inputs.

Checks cover finite strictly ordered means, positive finite mass, conserved
mass, true extrema, means within support, CDF bounds/monotonicity, quantile
support/endpoints/monotonicity, and unchanged bytes after queries. Integer and
binary-fraction cases require exact mass. Decimal/scaled inputs allow 64 machine
epsilons of relative accumulation error with no absolute epsilon. Compressed
digests are not asserted to match exact input ranks.

Exit 0 means the selected matrix completed and passed, exit 1 means a failure,
and exit 2 means the time budget prevented completion. The default budget is
20 seconds. Controls include `--seed`, `--trials`, `--max-size`, and
`--max-seconds`. Keep controls fixed when comparing revisions. This is not a
timing benchmark; reported elapsed time only documents the bounded audit.

## Scope corrections and files

`initial_configuration_report.json` records the initial setup mistake: the
Python API rejects capacities 1/2/3. These are now checked as expected validation
errors. A separate native-frontend attempt also rejected them; it establishes
no Rust-core numerical result. The exploratory `--surface native` option remains
available for reproducing that historical rejection, and is not a passing gate.
`historical_initial_audit.py` preserves the original script and its hash.

`recorded_public_result.json` contains the full two failures and exact input
arrays. `compressed_duplicate_reproducer.json` reduces one failure to 20 points
at capacity 19 and records its scale dependence and ambiguous duplicate-mean CDF.
Its coalesced-midpoint comparison describes the stored centroid representation,
not an exact-rank accuracy oracle for compressed input data.

All totals in this matrix are finite and normal. Commit `8ce82fc` adds separate
subnormal-total query regressions; this audit does not cover that path. Aggregate
overflow, broader distribution accuracy and additional seeds remain separate
work. Preserve the original failures and strict ordering assertion when comparing
revisions; do not overwrite the historical record with a passing result.
