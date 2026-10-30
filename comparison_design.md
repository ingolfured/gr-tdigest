# Why choose modern gr-tdigest over legacy G-Research?

For new applications, **Modern is our recommended engine**. It combines substantially better accuracy on the recorded real-data workloads with a shared Rust/Python/Polars/Java API, CDF queries, fractional weights and portable digest state. The original G-Research implementation remains available for applications that need its exact existing results.

The strongest reason to move is measurable: **Modern wins average rank error in 279 of 324 real-data cases, tail rank error in 295, and average value error in 260**, at matched logical centroid-field byte budgets. These are the results of the recorded comparison, including cases Modern loses.

This document explains that recommendation. [Full setup tables](compat/quality-20261020/RESULTS.md) show the winners, error ratios and precision breakdowns.

## 1. Better measured accuracy on recorded workloads

The suite uses every row of two public datasets: Bike Sharing and household Energy. It tests nine measured columns, f32/f64, sizes 32/128/512, natural/reversed input and three construction paths: single build, 40 sequential merges and a balanced 40-way merge.

Across the averages for all nine size/merge groups in each dataset:

| Workload | Average rank error | Tail rank error | Average value error | Peak value error |
| --- | --- | --- | --- | --- |
| Bike | **Modern wins** | **Modern wins** | **Modern wins** | **Modern wins** |
| Energy | **Modern wins** | **Modern wins** | **Modern wins** | GR wins |

The benefit is substantial at practical summary sizes. For a single build with Modern size 128:

| Dataset | Modern average rank error / GR | Modern average value error / GR | Modern peak value error / GR |
| --- | ---: | ---: | ---: |
| Bike | **0.017** | **0.035** | **0.35** |
| Energy | **0.043** | **0.40** | 1.84 |

**Lower error is better. Every ratio is Modern error divided by GR error: below 1 favors Modern; above 1 favors GR.** For Bike in this setup, Modern has about 1.7% of GR's average rank error and 3.5% of its average value error. For Energy, it has about 4.3% of GR's average rank error and 40% of its average value error.

The advantage continues through repeated merges. At size 128, Modern's grouped average rank error is about **0.021 times GR's on Bike** and **0.009 times GR's on Energy** for both sequential and balanced 40-way construction.

These are ratios of average per-case errors, averaged across each dataset's columns, both precisions and both orders. Value errors are normalized by each column's IQR, with the suite's fallback for zero IQR. Peak value error averages each case's maximum value error. Grouped wins do not imply every individual case wins.

## 2. A broader engine for new applications

Accuracy is one reason to choose Modern. Its API and state model also address work that otherwise belongs in application code.

| Need | What Modern provides |
| --- | --- |
| Percentiles and distribution queries | Quantile, median and CDF through the shared core; the preserved original GR API has no CDF method |
| Fractional observations and accumulated mass | Positive fractional weights, with f64 centroid weights in both mean precisions; legacy GR retains u32 centroid weights |
| One implementation across a stack | Rust, Python, Polars and Java surfaces with shared validation and cross-language coherence tests |
| Portable stored summaries | Versioned TDIG serialization, precision/configuration metadata and cross-language round-trip checks |
| Explicit control over precision and compression | f32/f64 means, precision casts, scale functions and singleton policies |
| Predictable frontend behavior | Explicit validation and rejection of incompatible merges instead of application-specific assumptions |
| Independent Python construction | Native construction releases the GIL after copying inputs into owned Rust storage |

Together, these make Modern our preferred foundation for new distribution-summary pipelines. The recommendation combines measured real-data accuracy with capabilities and integration; it is not a claim that every query is faster or every error metric is smaller.

## 3. Where legacy GR still does better

The main measured tradeoff on real data is **Energy peak value error**. Modern's grouped peak value error is **1.81–1.86 times GR's** across the tested sizes and merge paths, even while Modern wins average rank, tail rank and average value error in those groups. Applications driven by the largest value-space error should examine this result directly.

Synthetic grids also identify places to improve. Averaged across tested sizes, precisions, orders and construction paths, GR wins all four reported metrics on Gaussian, exponential, lognormal, Pareto, bimodal and signed-scale distributions. Modern wins all four on drift, skewed two-point and zero-inflated data; rounded ties and uniform grids split by metric. The [distribution tables](compat/quality-20261020/RESULTS.md#synthetic-data-winner-by-distribution) quantify these differences.

We retain these results because they make the recommendation useful: developers can check the error measure and data shape that matter to their application. They also give further accuracy work a concrete target.

## 4. Upgrade deliberately; preserve existing applications

Choose Modern explicitly:

```python
from gr_tdigest import TDigest
```

Keep legacy GR when reproducing existing numerical behavior is the requirement:

```python
from tdigest_rs import TDigest
```

The legacy interface uses pinned original calculations and the original Python wrapper. Its f32 arithmetic, u32 weights, immutable merge/update and copy/pickle behavior remain preserved. The original Rust interface is available through `tdigest_core`. The 91 unchanged upstream tests, independent exact-output fixtures and source checksums protect that compatibility.

Modern is a separate numerical engine, so matching compression parameters does not promise identical answers. See the [compatibility guide](compat/gr_tdigest_vs_tdigest_rs.md) for migration scope and [source provenance](src/upstream/PROVENANCE.md) for the retained implementation.

## 5. How the comparison earns its conclusions

Results come from [CI run 104](https://github.com/ingolfured/gr-tdigest/actions/runs/39566448129), which measured commit `e6a14a28237a77aca4df88955f3257b99a518d2d`. Modern uses K2/Use. Both engines receive identical observations after the same precision cast.

GR's compression parameter is selected by observed summary size, without consulting accuracy. Eligible comparisons fill 90–100% of Modern's logical centroid-field byte budget. All 324 real-data cases and 378 of 432 synthetic cases qualify. This budget counts centroid fields and excludes allocation overhead, capacity, RSS and wire size.

Independent sorted-data references measure rank, tail rank and percentile-value errors separately. CDF is checked against the empirical midpoint reference for Modern; there is no invented GR CDF comparison. The [test methods](tests/quality/README.md) define the references and gates.

The recorded regression gate passes with known limits: 10 synthetic cases exceed accuracy limits, and 13 of 15 separate numerical boundary controls fail. These remain visible in the results. The public fixtures establish behavior on these workloads; they are not independent held-out trials, a formal error guarantee or a throughput benchmark.

## 6. Design context beyond legacy GR

The direct measured comparison in this document is Modern versus original GR. Other projects inform individual design choices:

- [tdunning/t-digest](https://github.com/tdunning/t-digest) and the [t-digest paper](https://arxiv.org/abs/1902.04023) provide algorithmic context for scale functions and interpolation.
- [Apache DataSketches](https://datasketches.apache.org/docs/QuantilesAll/QuantilesOverview.html) provides context for formal rank guarantees and other quantile-sketch families.
- [Appian polars-tdigest](https://github.com/appian/polars-tdigest) provides Polars integration context.
- [Elasticsearch's t-digest implementation](https://artifacts.elastic.co/javadoc/org/elasticsearch/elasticsearch-tdigest/8.17.9/org.elasticsearch.tdigest/org/elasticsearch/tdigest/package-summary.html) provides host-engine integration context.

These references are design background, not a benchmark ranking. The case for adopting Modern rests on the measured GR comparison and the capabilities shipped here.

## Modern versus original GR: measured accuracy (CI, 21 October 2026)

For the complete size/merge/precision breakdown, see [recorded results](compat/quality-20261020/RESULTS.md). Every cell names the winner and uses the same Modern/GR error ratio. The [API contract](api_design.md) and [implementation design](tdigest_design.md) document the behavior behind those results.

These figures describe historical CI run 104 at `e6a14a2`. They compare accuracy at matched logical centroid-field byte budgets, excluding allocation overhead and RSS; they do not establish a throughput ranking or equal process memory use. Later measurement-input and regression-gate changes do not establish a new accuracy winner without fresh measurements. That run passed its regression gate while retaining 10 synthetic accuracy-limit failures and 13/15 separate numerical boundary failures.

## Default settings and practical accuracy (21 October 2026)

The original default is delta 300; modern defaults to capacity 1,000. Earlier
modern capacity 200 does not define the migration comparison. With the same
observations and controlled mean precision, the 396-profile default comparison
reduces average rank error by 98.7%, or 99.6% on its 108 recorded-data profiles.
These are descriptive averages over related profiles, with different storage.

At comparable logical field storage, modern capacity 1,000 improves average
rank error in 213 of 234 eligible established-data profiles and tail rank error
in 223, with 11 tail ties and no losses. Both comparisons retain numerical
value-error losses, and the query-level reconstruction explains observed-point
versus type-7 interpolation differences and local merge/compression tradeoffs.

The practical standard is no obvious, unexplained accuracy deterioration, with
separate scores and explanations. It is not a requirement to dominate every
measure on every input. Hard correctness, exact original compatibility and
frozen regression gates remain unchanged. Public claims should show the measured
gains prominently alongside their configuration, storage and domain.
See [the accuracy report](tests/quality/ACCURACY.md) and
[its source-bound evidence](compat/accuracy-20261021/).
