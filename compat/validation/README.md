# Validation of the K2Norm domain correction

Clean source `ff73cfab31d6368d203c1a361b8efaaea29d735b` passes Linux build,
test and lint: **334 Python/coherence + 91 untouched original cases**, **121
Rust tests**, and Java. Three pre-existing Rust cases remain ignored.
The optimized wheel passes **2,022 test instances** across seven environments:
six CPython 3.10–3.12 / NumPy 1–2 cells each run 208 compatibility + 90 modern
cases; a Polars-free cell runs 208 + 26. There are no skips or xfails in these
installed gates. An unmodified normalized source archive produces a distinct
wheel which passes its own same seven-cell gate.

The direct artifact also passes the unchanged 800-digest weighted-point and
1,440-route / 3,840-state compressed audits. The new 20-case K2Norm suite first
produced 14 expected failures and six passing controls on the preserved old
wheel, then all 20 passed on the fix. Focused Rust cases pass debug and release.
The original compatibility calculation path is unchanged.

See [canonical results](ff73cfa-validation.json),
[artifact provenance](ff73cfa-provenance.json),
[installed audits](ff73cfa-modern-wheel-validation.json), and
[raw logs, matrices and negative control](runtime-ff73cfa/).
These are pinned Linux runtime/artifact checks, not manylinux, cross-platform,
unconstrained resolver or universal accuracy guarantees. Earlier results retain
their source labels below.

# Validation of original Rust and Python runtime delivery

The library source at `360ad5b` passed the Linux canonical build/test/lint gates;
`3d0182a5713be0d1835296a8c7a340c987243072` changes only the artifact gate's
metadata formatting/order assertions and CHANGELOG. The canonical results are:
314 Python/coherence + 91 untouched original Python cases,
116 Rust tests and Java tests; three existing Rust tests remain ignored.

The actual optimized cp310 ABI3 wheel passed seven clean environments, totaling
1,902 test instances with no skips/xfails. Six cells cover CPython 3.10–3.12 ×
NumPy 1/2 with optional Polars installed; one CPython 3.10/NumPy 1 cell has Polars
genuinely absent. Every full cell runs 208 compatibility + 70 modern cases;
the Polars-free cell runs 208 + 26. Repeated environment cases are not distinct
test definitions. Versions and native/import hashes are retained per cell.

A genuine normalized sdist was rebuilt with `--locked`, without modifying any
extracted source file. Its distinct wheel passed its own 1,902-instance gate
and contains byte-identical native code to the direct wheel.
The direct wheel also passes the unchanged weighted-point and compressed-state
audits. The canonical command logs, both artifact identities, matrices and
source-file hashes are in [runtime-3d0182a](runtime-3d0182a/).

See [canonical results](3d0182a-validation.json),
[direct wheel provenance](3d0182a-provenance.json),
[additional installed audit commands](3d0182a-modern-wheel-validation.json),
[point audit](3d0182a-modern-weighted-point-audit.json) and
[compressed audit](3d0182a-modern-compressed-state-audit.json).
Original Rust crate/package/oracle evidence has its own
[source-labelled record](../numerical-review/original-rust/README.md).

These are Linux-native artifacts with explicitly staged/pinned dependency
combinations. They do not prove unconstrained dependency resolution, manylinux,
musllinux, macOS/Windows/ARM builds, Python 3.13+, or safe co-installation with a
second distribution owning the same package paths. Historical reports below
retain their original source and narrower support claims.

# Validation of safe scaling and corrected packaging

Clean source `52f6212008111b62fa07374278b850868b9ea77e` passed Linux `make build`,
`make test` and `make lint`: **314 Python/coherence cases plus 91 untouched
upstream cases**, 112 Rust tests and passing Java tests. Three pre-existing Rust
tests remain ignored. The optimized installed wheel passed all **208 compatibility
cases**, with no skips or xfails, and **70 modern API/regression cases**.

The same artifact passed the 800-digest weighted-point audit (33,536 CDF/rank and
304 integer-mass checks) and 1,440 compressed-state routes / 3,840 states.
These are bounded checks, not a proof for every finite input.

The new scaling change rejects invalid products/totals transactionally; it does
not cover input/merge aggregate overflow or weighted-value sum overflow. The
packaging changes exclude caches and remove a duplicate README archive path.
Actual standard source-archive rebuilding and same-distribution upgrade/removal
are documented in [package ownership](../package-ownership/README.md).

See [canonical commands and results](52f6212-validation.json),
[wheel provenance](52f6212-wheel-provenance.json),
[installed modern checks](52f6212-modern-wheel-validation.json),
[point audit](52f6212-modern-weighted-point-audit.json),
[compressed audit](52f6212-modern-compressed-state-audit.json), and
[restack equivalence](52f6212-restack-equivalence.json). All code, tests and build
configuration matched the validated source exactly after restacking; only
CHANGELOG ordering differed before adding these records and documentation.

Environment remains WSL Linux x86-64, Python 3.12.12, NumPy 2.3.4, Polars 1.34.0,
Rust 1.90.0 and JDK 21. No wider platform/Python support or public Rust source
compatibility is claimed. Performance results at `c5dec35` predate this batch.
Historical source-labelled validation follows unchanged.

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
