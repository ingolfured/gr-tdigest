# Validation of the numerical boundary fixes

Clean source `c5dec35059990b40fe6f036dc83590bf0cdf2a70` passed the canonical Linux
build, test and lint gates. The full suite includes **294 Python/coherence cases
plus 91 unchanged upstream cases**, Rust and Java. Three existing Rust tests stay
ignored; none of the 208 installed-wheel compatibility cases is skipped or xfailed.

The exact optimized wheel also passed **40 modern Python/Polars API cases**, the
**800-digest weighted-point audit**, and the formerly failing compressed-state
matrix: **1,440 routes, 3,840 states, zero failures**. The latter checks invariants,
not exact rank accuracy after compression. Both audits retain their original
controls and script hashes.

Wheel SHA256: `1e5e48b0789f2aa336b9d36b387b2997acfb2da9ed92bb98159dc01e04267f99`.
Native extension SHA256: `44247fa34031a74180a4ab3c8c0e019b7e5304af9a19e83cf7f249ba418c448c`.
Environment: ArchFresh WSL Linux x86_64, Python 3.12.12, NumPy 2.3.4,
Polars 1.34.0, Rust 1.90.0 and JDK 21.

See the [gate manifest](c5dec35-validation.json),
[wheel provenance](c5dec35-wheel-provenance.json),
[installed modern commands](c5dec35-modern-wheel-validation.json),
[point audit](c5dec35-modern-weighted-point-audit.json),
[compressed audit](c5dec35-modern-compressed-state-audit.json), and
[restack equivalence](c5dec35-restack-equivalence.json).
The three PRs were restacked after validation: all tracked source, tests and build
configuration matched the validated candidate exactly; only changelog ordering
differed before adding this evidence. The source commit is retained on the
`review/evidence-numerical-boundaries-20261014` branch for reproducibility.

These fixes do not cover compression underflow, every tiny component within a
normal total, aggregate/product overflow, or all malformed seeded state. They do
not establish original public Rust API compatibility, a safe package upgrade, or
the full manylinux/platform release matrix. Historical records follow unchanged.

# Historical validation at source 22867bd

These records identify the exact committed source and optimized Linux wheel used
for the completed gates. They are separate from the Rust-core timing evidence in
[performance](../performance/README.md).

Source: `22867bd55739a96a6fab3c8c48b27406ecb32197` (clean at validation).
Wheel SHA256: `66cce5d5483ae05151bca7f7b55383f01578c66b0ad8bbdd4a1b2983583f3869`.
Environment: ArchFresh WSL Linux x86_64, Python 3.12.12, NumPy 2.3.4,
Polars 1.34.0, Rust 1.90.0 and JDK 21.

| Gate | Recorded outcome |
|---|---|
| `make build` | Passed |
| `make test` | 284 Python/coherence cases and 91 unchanged upstream cases; Rust and Java passed |
| `make legacy-wheel-test` | 208 installed-wheel cases: 91 unchanged upstream, 70 exact numerical, 13 API, 2 optional import, 32 genuine published pickle restores |
| `make lint` | Passed |
| Additional installed modern Python/Polars API checks | 40 passed outside the source checkout |
| Installed-wheel weighted point audit | 800 digests, 13,568 CDF checks, 19,968 quantile-rank checks and 304 integer-mass checks; zero ordinary failures |

See the [gate manifest](22867bd-validation.json), [wheel provenance](22867bd-wheel-provenance.json),
[additional installed-wheel commands](22867bd-modern-wheel-validation.json), and
[weighted-point audit output](22867bd-modern-weighted-point-audit.json).
Absolute paths in these raw records are historical provenance, not portable
installation instructions. Wheel/native hashes connect the records; a source
directory alone does not identify the loaded extension. The wheel binary and
full local logs are not committed to Git.

## Reproduction and limits

On a Linux checkout of that revision with the documented project dependencies,
run `make build`, `make test`, `make legacy-wheel-test` and `make lint`.
The installed-wheel gate builds and tests an optimized artifact in isolation.
For the independent weighted audit, use the commands and reference definitions in
[its README](../weighted-point-audit/README.md) with a Python environment containing
the built wheel. At historical `22867bd`, the optional extreme probes were known failures outside
its passing ordinary gate. The query-only subnormal fix at `c5dec35` has separate
regressions; compression underflow and aggregate overflow remain open.

The 208 compatibility checks had no skips, xfails or weakened upstream assertions.
Loading genuine old pickles emitted 16 NumPy deprecation warnings. This validates
the tested `--compatibility linux` artifact; it does not establish manylinux
compliance or every supported platform. No original public Rust API compatibility
is claimed. Subnormal-mass and overflowing-total limits remain documented, and the
uncompressed point audit is not a general compressed-data accuracy guarantee.

A subsequent [compressed-state invariant audit](../compressed-state-audit/README.md)
failed on two strict-order cases involving adjacent f64 means and large finite
weights. Its failing reproductions are retained separately. Passing the historical `22867bd` gates did not resolve that defect. The later
`c5dec35` fix and passing audit rerun are recorded at the top of this document.
