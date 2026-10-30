# Numeric representability evidence

These are unchanged historical records from 21 October 2026 for
source `aaf7ab3956fed4269c1eaeef915a6faee78fa305` and the
`0.2.5.dev2026102102` / `0.2.5-dev2026102102` artifacts.
They do not qualify current `0.3.0` binaries.

This deterministic Float32/Float64 sweep records 204 constructed states,
six constructor rejections, 1,692 successful transforms and 348 explicitly
rejected transforms, with zero unexpected failures. Rejections remain
visible and are not counted as successful construction or transformation.

The 1,686-line JSON is the unchanged raw observation table for one sweep.
Review the totals, each rejection and recorded source/native identity;
this file adds no code, supported domain, fixture or acceptance threshold.
It describes the old artifact in [ARTIFACTS.md](ARTIFACTS.md). The sweep
runner is not included here, so this is retained evidence rather than a
new executable release gate.

Files:

- [numeric-limits-optimized.json](numeric-limits-optimized.json)
