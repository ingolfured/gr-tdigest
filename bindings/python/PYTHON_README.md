# gr-tdigest (Python)

Python packaging metadata for `gr-tdigest`.

Install `gr-tdigest` for the NumPy APIs, or `gr-tdigest[polars]` for Polars expressions.
The Polars extra currently pins the tested version 1.44.2.
Supported runtimes are standard GIL builds of CPython 3.10–3.15.
Python 3.10–3.12 supports NumPy 1.26.4 or 2.x. Python 3.13–3.14 requires
NumPy >=2.3.4,<3, and Python 3.15 requires NumPy >=2.5.4,<3.
The repository runtime gate records the exact tested dependency combinations.

For full project documentation, examples, and release notes, see the repository root `README.md`.

The optional Polars integration is pinned to1.44.2, matching the Rust plugin ABI. Source builds use the pinned Rust1.95 toolchain. The Linux release gate installs the same ABI3 wheel on3.10–3.15, then rebuilds the actual source archive with3.12,3.14 and3.15. These claims cover standard CPython; free-threaded variants and PyPy are not qualified.
