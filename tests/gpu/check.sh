#!/usr/bin/env bash
# Run from the repository root on a Linux machine with HIP, Rust, uv and a JDK.
set -euo pipefail
: "${HIP_PLATFORM:?Set HIP_PLATFORM to amd or nvidia}" "${GPU_ARCH:?Set GPU_ARCH to the device architecture}"
case "$HIP_PLATFORM" in
    amd) flags=(-x hip -ffp-contract=off "--offload-arch=$GPU_ARCH") ;;
    nvidia) flags=(-x cu --fmad=false --ftz=false "-arch=$GPU_ARCH" -D__HIP_PLATFORM_NVIDIA__
                   "-I${CUDA_PATH:-/usr/local/cuda}/include" "-I${ROCM_PATH:-/opt/rocm}/include") ;;
    *) echo "HIP_PLATFORM must be amd or nvidia" >&2; exit 2 ;;
esac
export GR_TDIGEST_TEST_DEVICE="${GR_TDIGEST_TEST_DEVICE:-0}"
out="$PWD/build/hip-check"
mkdir -p "$out/classes"
wheels=$(mktemp -d "$out/wheels-XXXXXX")
export UV_PROJECT_ENVIRONMENT="$out/env"
uv sync --locked --project bindings/python --all-groups --no-install-project --python 3.12
env_bin="$UV_PROJECT_ENVIRONMENT/bin"
(cd bindings/python && "$env_bin/maturin" build --release --locked --features python,hip \
    --compatibility linux --interpreter "$env_bin/python" --out "$wheels")
uv pip install --reinstall --no-deps --python "$env_bin/python" "$wheels"/*.whl
"$env_bin/python" -m pytest -q -o addopts='' tests/gpu
cargo test --release --locked --features hip --lib merge_workspace_budget_counts_every_device_buffer
cargo test --release --locked --features hip --test gpu_native -- --nocapture
"${ROCM_PATH:-/opt/rocm}/bin/hipcc" -std=c++17 "${flags[@]}" tests/gpu/training_native.hip -o "$out/training-native"
"$out/training-native" "$GR_TDIGEST_TEST_DEVICE"
cargo build --release --locked --features java,hip
target_dir=$(cargo metadata --no-deps --format-version=1 | "$env_bin/python" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')
javac --release 11 -d "$out/classes" bindings/java/src/gr/tdigest/*.java tests/gpu/GpuSmoke.java
java -ea -Dgr.tdigest.native="$target_dir/release/libgr_tdigest.so" -cp "$out/classes" GpuSmoke "$GR_TDIGEST_TEST_DEVICE"
