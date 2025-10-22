# tdigest-rs

A Polars plugin and Rust library for distributed quantile estimation using [T-Digest](https://docs.rs/tdigest/latest/tdigest/).

- Fast, mergeable quantile estimation for large/distributed datasets
- Native Rust implementation, with Python bindings for Polars
- Supports both canonical (f64) and compact (32-bit) payloads
- Easily integrates with Polars DataFrames and expressions

## Example

See the [Yellow Taxi Notebook](./tdigest_yellow_taxi.ipynb) for a usage example.

Minimal Python usage:
```python
from tdigest_rs import tdigest
import polars as pl

df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
df = df.with_columns(
    tdigest("values", max_size=100, use_32=True)
)
```

## Requirements

- **Python:** 3.12+ recommended (works with 3.8+)
- **Rust:** 1.70+ (2021 edition)
- **Polars:** 0.51.x (Rust and Python)
- **Maturin:** for building Python extensions

## Development Setup

1. **Python environment:**
   ```bash
   python3.12 -m venv .env
   source .env/bin/activate
   python -m pip install -r requirements.txt
   ```

2. **Rust toolchain:**
   - Install from [rustup.rs](https://rustup.rs/)
   - Recommended: `rustup update stable`

3. **Build the Python extension:**
   ```bash
   maturin develop
   # or for optimized builds:
   maturin develop --release
   ```

4. **Pre-commit hooks:**
   - This repo uses [pre-commit](https://pre-commit.com/) to enforce formatting, linting, and test hygiene.
   - Install pre-commit and set up hooks:
     ```bash
     pip install pre-commit
     pre-commit install
     ```
   - To run all checks manually:
     ```bash
     pre-commit run --all-files -v
     ```
   - The hooks will:
     - Auto-fix Python with Ruff
     - Run `cargo build`, `cargo fmt`, `cargo clippy --fix`, and `cargo test` (all)
     - Fail fast on the first error

## Rust Development

- To build and test with Cargo:
  ```bash
  cargo build
  cargo test
  ```

## Python API

- The plugin exposes a `tdigest` function for Polars expressions:
  ```python
  from tdigest_rs import tdigest

  # Use use_32=True for compact 32-bit payloads
  df.with_columns(
      tdigest("col", max_size=100, use_32=True)
  )
  ```

## Versioning

- All Polars crates are pinned to the same version (`0.51.x`) for compatibility.
- Python requirements are in `requirements.txt` and support Python 3.12+.
- See `Cargo.toml` and `pyproject.toml` for details.

## Contributing

- Please run all pre-commit hooks and ensure all tests pass before submitting a PR.
- For Rust code, follow `cargo fmt` and `cargo clippy` suggestions.
- For Python code, follow Ruff and type-check with `mypy`.
