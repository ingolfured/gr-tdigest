# Installed consumer evidence

These are unchanged historical records from 21 October 2026 for
source `aaf7ab3956fed4269c1eaeef915a6faee78fa305` and the
`0.2.5.dev2026102102` / `0.2.5-dev2026102102` artifacts.
They do not qualify current `0.3.0` binaries.

The exact old JAR is exercised through Scala/Kotlin consumers using a
controlled authenticated Maven fixture, wrong-credential rejection and a
stopped-repository offline rerun. This fixture does not qualify internal
build services. Expected HTTP 401/404 and nonzero negative-test exits stay
in the evidence.

Installed Python checks record typing, public 0.2.4 upgrade/rollback,
import ownership and uninstall behavior. Toy Float32/Float64 states have
Python/JVM byte/query parity; PyArrow 24/Polars 2.0 struct IPC/Parquet
round trips pass. These are the artifacts in [ARTIFACTS.md](ARTIFACTS.md),
not a fresh validation of current consumer APIs.

Files:

- [languages-story.json](languages-story.json)
- [python-extras-story.json](python-extras-story.json)
- [installed-extra-story.json](installed-extra-story.json)
