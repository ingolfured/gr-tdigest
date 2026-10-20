# Python runtime validation, 20 October 2026

One unchanged CPython3.10 ABI3 wheel passed nine full installed environments across CPython3.10–3.15 and two Polars-free environments (3.10/3.15). Each full environment ran208 original/API compatibility cases and184 modern cases. Each Polars-free environment ran208+120. Tests ran with isolated interpreters outside the checkout, checked installed import paths/native hashes and required clean dependency checks and no skipped/failed cases.

The actual verified source archive was independently rebuilt using CPython3.12,3.14,3.15, with no version overrides or retagging. Each rebuilt wheel passed392 installed cases. All three rebuilds produced the same native-library hash. There were5,360 passing test-case executions across these gates.

`make build test`, `make lint`, original-source checksums and all91 original tests passed. The old original numerical crate and Python wrapper remain unchanged.

The dependency update is coupled: PyO3 0.29.3, rust-numpy0.29, pyo3-polars0.28, Polars Rust0.55.2/Python2.0.0. The optional Polars extra therefore moves from1.34 to2.0. Source builds require the pinned Rust1.95 compiler. Standard CPython on Linux x86_64 is qualified here; free-threaded builds, PyPy and other operating systems need separate qualification.

[Machine-readable evidence](validation.json) contains source/artifact identities, every runtime cell, source rebuilds and test-output hashes. The release workflow runs this same gate against its actual wheel and source archive before publication.
