# Checks in the consuming environment

Private wheels, consumer fixtures, internal services and the actual executor
image stay in the intranet. External development does not need access to them.
This guide supplies runnable public checks; the owner runs the private checks
and decides whether the candidate is suitable for deployment.

## Keep artifact and machine identities

Install the exact candidate wheel from its manifest into a fresh environment
using the documented package manager and internal dependency source. Do not
rebuild under the same version. Keep the wheel/JAR SHA-256 values alongside the
results. In that environment, record its runtime, installed native payload,
JAR and machine:

```sh
python -I scripts/qualify_host.py --jar /staged/gr-tdigest-CANDIDATE.jar \
  --output /results/candidate-host.json
```

The receipt records CPU affinity, available physical/logical cores, cache and
NUMA topology, memory, visible cgroup quotas and native pool settings. An
unavailable quota stays unknown. Optional `--require-physical-cores N` exits 2
and records `UNTESTED_INSUFFICIENT_CORES` when that particular check cannot be
performed. Merely recording a host never qualifies its workload or performance.

## Compare the same workloads

Use a separate environment for each baseline and the candidate, without other
heavy work running. Run the same benchmark script, input sizes, CPU affinity,
worker counts and native-pool settings. Retain its script hash. The source label
identifies the library source, not the check script; the native hash identifies
the loaded binary. For example:

```sh
export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I scripts/consumer_benchmark.py --source-sha LIBRARY_SOURCE_SHA \
  --worker concurrency --workers 1 2 4 8 16 32 64 100 \
  --tasks-per-batch 400 --output /results/candidate-concurrency.json
```

Choose counts for the actual server; requesting 100 workers on a small host
does not qualify a 100-CPU server. Every batch must contain at least as many
tasks as workers. Reports include throughput, submission-to-completion tail
latency, process CPU time, average CPU cores actually used and peak process
memory. Independent digests are used; this is not concurrent mutation of one
handle. These are short bounded workloads, not a universal speed or memory gate.

Omit `--worker` to run all scenarios in fresh processes: thousands of groups,
16/64/128 columns, fragmented inputs, 40-way merging, bounded construction,
serialization/query bridges and independent-digest concurrency. Use symmetric
baseline/candidate run order and repeated measurements. Agree representative
inputs and acceptable tradeoffs before sign-off. A baseline wrapper may need an
explicit adapter if its API differs; do not silently substitute its engine.

The intranet 0.1.9/0.1.11 wheel regression checks remain owner-run. Keep their
existing exact/tolerance gates and fixtures. A public source-built comparison
cannot substitute for those binary baselines or establish their missing native
build provenance. Do not relax or regenerate consumer fixtures to make an
upgrade pass.

## Cluster and internal build services

Use the [Spark check](../../bindings/java/examples/spark/README.md) with the
actual shared checkpoint URI, executor image and selected serializer settings.
Its local separate-executor result is useful public evidence, not the final
cluster result. Confirm the required Spark/Scala versions and configuration.

Use the [JVM bootstrap/build guide](../../bindings/java/README.md) to stage
approved toolchains and dependencies from internal sources. Verify configured
authentication/trust, unavailable-source failure without public fallback and
an offline build after staging. Resolution and publication destinations stay
separate. The external authenticated-fixture and stopped-repository checks do
not qualify the real internal services. Publication remains an owner decision.

Retain passed, failed and untested cells with exact commands and artifact hashes.
Neither an unavailable private input nor an unavailable large server is a
passing cell. No external access request is needed to keep these checks pending.
