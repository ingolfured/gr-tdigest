# Follow-up checks: 21 October 2026

These receipts qualify the exact development wheel and JAR from
**aaf7ab3956fed4269c1eaeef915a6faee78fa305**, version
Python `0.2.5.dev2026102102` / JVM `0.2.5-dev2026102102`.
The subsequent Spark check and owner-run tools are separate verification source;
they do not change the library or the artifact bytes. No production release is
published or approved by these results.

## Findings

| Question | Observed result | Consequence |
| --- | --- | --- |
| Scaling function | Alternate scales shift central, value and tail errors differently. Fixed capacity does not imply identical actual storage. | Scaling is a major cause of the tradeoff; it is not proof that all losses are unavoidable. Keep the selected default. |
| Capacity 16 | Deliberate small-summary stress setting. All 114 diagnostic misses are at capacities 16/32/64; none occur in the sampled 200/1000 profiles. | Keep stress evidence visible and judge practical capacities separately. No universal accuracy guarantee follows. |
| Forty summaries | Forty input shards each build a digest; those digests are combined sequentially or as a balanced tree. | This models modest merge fan-in, not a 40-centroid configuration. |
| Performance | Up to 28 full trial compressions were discarded after reading their lengths. Count-only trials preserve the emitting compression and exact decisions. | Exact-wheel streaming takes 16.5% less time and 40-way merging 9.3% less time than the previous candidate. |
| Earlier public modern baseline | Against source 1451f94, the same paths take 5.7% and 4.4% less time. | This is a bounded modern-to-modern comparison, not an original-engine throughput claim. Memory improvement is not established. |
| Numerical edge cases | All 15 exact controls pass. The representability sweep constructs 204 states and rejects six overflow cases; 1,692 transforms succeed and 348 reject explicitly. No unexpected failure or failed-operation mutation was observed. | Supported state remains valid; unsupported numeric ranges fail explicitly. |
| Distributed state | Four exact-JAR Spark cells pass with two executor processes. | See the [Spark receipts](../spark-20261021/README.md); the real intranet image/configuration remains owner-run. |
| Available-core scaling | At 1/2/4/8/16/32 workers, the bounded workload reaches about 53/104/200/363/521/450 million values/s. Host: 16 physical, 32 logical, one NUMA node; visible quota unavailable. | Additional logical workers do not guarantee more throughput. This does not qualify a 100-plus-CPU server. |
| Private artifacts/services | They remain inside the intranet. Controlled external authentication, bad-credential and prepared-cache offline checks pass. | Use [owner-run checks](../../tests/quality/OWNER_CHECKS.md); no private access is requested. |
| Historical SSD source | The supplied mount is unavailable externally. | Comparison remains pending; historical source/evidence refs are retained. |

## Combined artifact evidence

- `make build`, `make test`, `make lint` and denied-warning Rust documentation
  checks pass after the performance change and after the verification-tool changes.
- The exact wheel passes 12 installed runtime/dependency cells and three source
  builds, totalling 6,285 repeated test executions. The receipt enumerates exact
  Python/NumPy/Polars versions; it is not a blanket NumPy 2.x claim.
- The same JAR passes Java 11/17/21 tests and actual glibc 2.17 / Java 11 execution.
  The wheel passes 433 installed tests on actual glibc 2.28. The native binaries
  have separate ABI floors and hashes.
- Scala 2.12.16 and Kotlin consumers resolve the exact JAR through a controlled
  authenticated repository, reject wrong credentials with no public fallback,
  then run offline with the repository stopped.
- Installed typing, import ownership, upgrade/rollback, cross-language byte
  identity and PyArrow 24 struct/Parquet round trips pass.
- All 4,140 differential diagnostic state hashes match. The canonical packaged
  wheel independently reproduces the exact diagnostic rows. The existing 114
  diagnostic and ten extended-suite accuracy misses remain visible; no gate or
  fixture was relaxed. Original source, wrapper, independent oracle and exact
  fixtures remain unchanged.

Read `state-and-metric-parity.json`, `packaged-performance-summary.json`, the
installed matrix and the exact-artifact qualification receipt for identities
and scope. Timing runs use symmetric order, four pinned physical cores and
controlled native pools, with no concurrent task-owned builds/timings. The
concurrency sweep is a separate bounded run on the full available host.
The qualification receipt's `manifest.json` hash identifies its initial build
snapshot; the final delivery manifest preserves that snapshot separately.
Full raw logs and results are retained in the downloadable evidence archive.

The approximate comparison contains both gains and losses against the pinned
original at measured storage budgets. Neither the passing exact controls nor
absence of sampled capacity-200/1000 misses proves universal all-metric
superiority. Final private regression, actual cluster, internal service and
representative-server checks remain with the owner.
