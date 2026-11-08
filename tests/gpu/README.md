# Linux GPU backend

Build the wheel from `bindings/python` using `maturin build --release --locked --features python,hip --compatibility linux` with `HIP_PLATFORM` and `GPU_ARCH` set for your device. The CPU packages build without HIP.

```python
import numpy as np
from gr_tdigest.gpu import DeviceArray, GpuDigest, from_array

values = DeviceArray(np.array([1., 2., 3.]), device=0)
digest = from_array(values, max_size=100)
probabilities = digest.cdf_device(values)
quantiles = digest.quantile_device(probabilities).to_numpy()
merged = GpuDigest.merge_all([digest, digest])
```

To run Python, Polars, native Rust, compressor and Java/JNI checks, install HIP with hipCUB, the vendor toolkit and driver, Rust, uv and a JDK. From the root, run:

```sh
HIP_PLATFORM=amd GPU_ARCH=gfx1100 tests/gpu/check.sh
```

For NVIDIA set `HIP_PLATFORM=nvidia`, `CUDA_PATH` and the device architecture (for example `GPU_ARCH=sm_89`). The CUDA 12.8.1 build was checked for sm_89. NVIDIA hardware validation is pending. `GR_TDIGEST_TEST_DEVICE` selects the ordinal. Set `GR_TDIGEST_SECOND_DEVICE` for cross-device rejection tests. GitHub Actions runs the CPU checks.

With the HIP wheel installed, run `python tests/gpu/bench.py --device 0` for CDF, `bench_training.py` for construction and `bench_merge.py` for 40-child merges. Each compares completed calls to the CPU path.

Use `python tests/gpu/bench_groups.py --device 0` to time 2,500 independent groups of 40 prebuilt digests on CPU and GPU. Each timed output is checked against the CPU left fold. Add `--serial-gpu` to include the longer per-group GPU baseline.
