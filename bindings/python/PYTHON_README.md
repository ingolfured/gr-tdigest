# gr-tdigest (Python)

Python packaging metadata for `gr-tdigest`.

Install `gr-tdigest` for the NumPy APIs, or `gr-tdigest[polars]` for Polars expressions.
The Polars extra currently pins the tested version 1.44.2.
Supported runtimes are standard GIL builds of CPython 3.10–3.15.
Python 3.10–3.12 supports NumPy 1.26.4 or 2.x. Python 3.13–3.14 requires
NumPy >=2.3.4,<3, and Python 3.15 requires NumPy >=2.5.4,<3.
The repository runtime gate records the exact tested dependency combinations.

For full project documentation, examples, and release notes, see the repository root `README.md`.

The optional Polars integration supports 1.x and 2.x (`polars>=1.0.0`, with no upper cap), with development pinned to 1.44.2. Both ordinary and extra installs preserve an existing compatible Polars version. The same compiled wheel is tested from Polars 1.0.0 through representative 1.x releases and 2.0.0 across the supported Python versions. Future releases are allowed by the metadata and require testing before compatibility can be guaranteed. Source builds use the pinned Rust1.95 toolchain. The Linux release gate compiles the verified source archive once with Python 3.10, then installs that unchanged ABI3 wheel on every supported runtime from 3.10 to 3.15. These claims cover standard CPython; free-threaded variants and PyPy are not qualified.
