# Candidate artifact provenance

These are unchanged historical records from 21 October 2026 for
source `aaf7ab3956fed4269c1eaeef915a6faee78fa305` and the
`0.2.5.dev2026102102` / `0.2.5-dev2026102102` artifacts.
They do not qualify current `0.3.0` binaries.

The initial build snapshot and later combined qualification describe source
`aaf7ab3956fed4269c1eaeef915a6faee78fa305`, Python `0.2.5.dev2026102102`
and JVM `0.2.5-dev2026102102`. The initial snapshot correctly leaves JVM
qualification pending. The later receipt records actual glibc execution,
JAR/native identities and an explicitly failing wider accuracy diagnostic.
Its combined status retains numerical acceptance failure.

The installed artifact metadata records the old Polars 2.0 pin. It is not
the current package dependency contract. Command paths and log hashes are
historical references; the raw logs and binaries are not supplied here.

Files:

- [artifact-qualification-story.json](artifact-qualification-story.json)
- [build-stage-manifest.json](build-stage-manifest.json)
- [installed-matrix-artifact.json](installed-matrix-artifact.json)
