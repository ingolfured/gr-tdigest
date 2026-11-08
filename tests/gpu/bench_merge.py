"""Time the same 40-child left fold on CPU and on resident GPU digests."""

import argparse
from statistics import median
from time import perf_counter_ns

import numpy as np
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
for precision in ("f32", "f64"):
    rng = np.random.default_rng(414)
    children = [TDigest.from_array(rng.normal(size=1000), precision=precision, max_size=100) for _ in range(40)]
    resident = [GpuDigest(d, args.device) for d in children]
    assert GpuDigest.merge_all(resident).to_bytes() == TDigest.merge_all(children).to_bytes()
    times = [[], []]
    for i in range(5):
        for j in (i % 2, 1 - i % 2):
            start = perf_counter_ns()
            if j:
                GpuDigest.merge_all(resident)
            else:
                TDigest.merge_all(children)
            times[j].append((perf_counter_ns() - start) / 1e6)
    print(f"{precision}: CPU {median(times[0]):.3f} ms, HIP {median(times[1]):.3f} ms", flush=True)
