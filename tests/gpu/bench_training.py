"""Median end-to-end construction time for one digest with resident GPU inputs."""

import argparse
from statistics import median
from time import perf_counter_ns

import numpy as np
from gr_tdigest import TDigest
from gr_tdigest.gpu import DeviceArray, from_array

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
for size in (1000, 10_000, 32_000):
    xs = np.random.default_rng(143).normal(size=size)
    resident = DeviceArray(xs, args.device)

    def cpu():
        return TDigest.from_array(xs, max_size=100)

    def gpu():
        return from_array(resident, max_size=100)

    assert gpu().to_bytes() == cpu().to_bytes()
    times = [[], []]
    for i in range(5):
        for j in (i % 2, 1 - i % 2):
            start = perf_counter_ns()
            (cpu, gpu)[j]()
            times[j].append((perf_counter_ns() - start) / 1e6)
    print(f"{size} observations: CPU {median(times[0]):.3f} ms, HIP {median(times[1]):.3f} ms")
