# Quality checks

The suite measures specific errors. It does not turn them into one flattering score. The older heuristic quality score remains a historical test helper and is not an acceptance rule here.

## Routine checks

`make test` includes `make quality-fast`:small reference/weighted tests and42 comparison profiles. The real-data part samples512 evenly spaced observations per column. The distributions cover constant values, ties, zero-heavy data, separated groups, drifting values, ordinary bell-shaped data and long tails. They are deterministic grids, not claims about the quality of a random-number generator.

The weighted controls use exact rational arithmetic for the input masses. They cover all three singleton policies, actual f32/f64 storage, decimal/binary fractions, large piles, subnormal through very large weights, serialization and query ordering.

`make quality-extended` reads every row of both real datasets and runs756 profiles:21 data columns/distributions ×2 precisions ×3 sizes ×2 orders ×3 construction/merge paths. The paths are a single build,40 sequential merges and a balanced40-way merge. It currently returns **failure** for ten size32 synthetic merge profiles. See [recorded findings](../../compat/quality-20261020/RESULTS.md).

`make quality-boundaries` runs15 hand-checkable Rust controls. Current integration
work passes all15 without changing the expected values; the historical baseline
has13failures. They cover tiny-capacity mass, small tail contributions,
cancellation, fractional-weight medians and subnormal exact-point means. This
boundary result is separate from the ten approximate-accuracy failures.

## Regression checks with known limits

`make quality-regression` runs the complete comparison and boundary audit against frozen measurements and an explicit known-failure inventory. It may pass the **regression** check while reporting `accuracy_status=FAIL` and `status=KNOWN_LIMITS`. It does not turn an unresolved defect into a passing accuracy result. CI stores the reports even when a check fails. Added, removed or changed approximate-accuracy findings require review;
improvements should update that inventory after being examined. The exact boundary
audit accepts a known failure becoming a verified exact pass against the same
target; it rejects changed targets, missing/duplicate/new controls, incorrect PASS
labels and new or changed failures. Historical boundary measurements stay intact.

The frozen per-profile gate permits no relative error deterioration. Rank metrics allow one observation of resolution; other normalized errors allow2e-6 for f32 or1e-12 for f64. Each metric is checked independently. Dataset/protocol hashes, duplicate/missing cases and unexpected exceptions are also checked. Changing a baseline requires reviewed fresh evidence, not regeneration to hide a failure.

## What each measurement means

- **Rank error:** distance from the requested probability to the observed-data rank interval of the returned value. Correct answers at repeated values get an interval, rather than an artificial tie penalty. Mean, maximum, median and tail errors stay separate.
- **Value error:** difference from a declared type7 sample percentile, both in original units and divided by the middle-half data spread. If that spread is zero, use the full range, then1 for constant data. This is a specified percentile convention, not the only possible convention for discrete samples.
- **CDF error:** difference from the exact empirical midpoint probability at135 fixed percentile-based probes (deduplicated). This is a sampled check, not a supremum/KS guarantee. The original G-Research API has no CDF method, so its CDF fields are unavailable, never fabricated.
- **State checks:** finite/positive weights, preserved total mass and weighted contribution, ordered centroids/queries, input bounds, actual storage precision and exact state round trips. Original rounding/order diagnostics are recorded separately and do not remove inconvenient comparison outcomes.

## Fair comparisons and limits

Both engines receive identical observations after the same precision cast. The original source/wrapper remain checksum-bound. A count-only search chooses the largest observed original summary within the modern summary's logical field-byte budget. It never chooses a parameter by accuracy. The search trace and actual sizes are recorded; it is not a proof of a globally optimal original setting.

Modern logical bytes count mean(4/8)+weight8+kind1 per centroid; original bytes count mean(4/8)+weight4. This excludes headers, padding, spare capacity, allocation and RSS. Modern wire length is recorded separately. Original states filling less than90% or exceeding100% of that field budget remain in the results but are excluded from matched-budget win/loss totals. Accuracy-limit failures **remain** in those totals when both measurements exist.

These are empirical fixtures on two related domains, not a universal accuracy guarantee. Reversed single-build cases can be identical, and many profiles share data. The counts are test coverage, not independent statistical trials or confidence levels. No throughput ranking is inferred from test duration.

## Wider capacity/scale diagnostics

`make quality-budgets` is a separate slower4140-profile measurement:3276on the
established full-data corpus plus864on12additional deterministic shapes. It tests
K2capacities16/32/64/128/200/512/1000and selectedQuad/K1sizes, actualf32/f64,
input reversal and40-way merge paths. Modern/original comparisons atK2/200and1000
reuse the actual logical-byte budget search. Original CDF remains unavailable.

The current source has114failed diagnostic profiles across the complete matrix,
including newly exposed small-capacity/shape failures. The runner exits nonzero;
these are not waived, skipped or disguised as completed accuracy fixes. The old
756-profile frozen baseline and its ten-case inventory are unchanged. Profiles
share data and are not independent trials. See the consumer accuracy report for
per-metric tradeoffs, original win/loss counts and reproduction instructions.
