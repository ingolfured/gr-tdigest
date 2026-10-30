# Performance and host-scaling evidence

These are unchanged historical records from 21 October 2026 for
source `aaf7ab3956fed4269c1eaeef915a6faee78fa305` and the
`0.2.5.dev2026102102` / `0.2.5-dev2026102102` artifacts.
They do not qualify current `0.3.0` binaries.

The counterbalanced exact-wheel comparison records streaming and 40-way
merge timings, native identities and peak-RSS ranges. The earlier public
modern baseline is source `1451f94`; the prior and optimized artifacts
remain distinct. These are bounded comparisons, not original-engine
throughput rankings or established memory improvements.

A separate old-script concurrency run records configured 1/2/4/8/16/32
workers, CPU time and throughput on a 16-physical/32-logical-core host.
It did not count observed worker threads; the newer benchmark does. The
script hash distinguishes those protocols. Visible cgroup ancestors record
`max`, while the effective complete hierarchy is not established.

The 100-physical-core request remains `UNTESTED_INSUFFICIENT_CORES`. Neither
a recorded host nor configured workers qualify a 100-CPU server, production
load or private consumer workload. Native/JAR hashes refer to the old
candidate in [ARTIFACTS.md](ARTIFACTS.md).

Files:

- [packaged-performance-summary.json](packaged-performance-summary.json)
- [concurrency-owner-ready.json](concurrency-owner-ready.json)
- [host-owner-ready.json](host-owner-ready.json)
- [host-insufficient-100.json](host-insufficient-100.json)
