# Linux HIP tests

Build from `bindings/python` with `HIP_PLATFORM=amd GPU_ARCH=gfx1100 maturin build --release --locked --features python,hip --compatibility linux` and install that wheel in an environment with NumPy and pytest. For NVIDIA choose `HIP_PLATFORM=nvidia`, a supported CUDA toolkit, HIP headers/toolchain and the matching CUDA GPU architecture (for example `GPU_ARCH=sm_75`). Both use the same kernel source. Normal CPU builds omit the `hip` feature.

Run `python -m pytest -q tests/gpu` against the installed GPU wheel. Select a device with `GR_TDIGEST_TEST_DEVICE`. Run `python tests/gpu/bench.py --device 0` to compare full Python calls, and `cargo test --release --features hip --test gpu_native -- --nocapture` for the native Rust comparison. A missing GPU or driver fails the explicit GPU suite. NVIDIA hardware results are pending.
