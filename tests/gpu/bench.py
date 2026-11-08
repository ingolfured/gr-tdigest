"""Time completed CDF calls, including probe upload and host result download."""

import argparse
import json
import time
from statistics import median

import numpy as np
from gr_tdigest.gpu import GpuDigest
from test_cdf import assert_same, inputs

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=0)
p.add_argument("--rounds", type=int, default=9)
args = p.parse_args()
for precision in ["f32", "f64"]:
    for cap in [32, 277, 1000]:
        d = inputs(precision, cap)
        start = time.perf_counter_ns()
        g = GpuDigest(d, args.device)
        upload = (time.perf_counter_ns() - start) / 1e6
        for count in [1, 256, 4096, 65536, 1000000]:
            probes = np.random.default_rng(43).normal(size=count)
            assert_same(g.cdf(probes), d.cdf(probes))
            cpu, gpu = [], []
            for i in range(args.rounds):
                ops = [(cpu, d.cdf), (gpu, g.cdf)]
                for sample, call in ops[:: 1 if i % 2 else -1]:
                    start = time.perf_counter_ns()
                    call(probes)
                    sample.append((time.perf_counter_ns() - start) / 1e6)
            print(
                json.dumps(
                    dict(
                        precision=precision,
                        cap=cap,
                        device=args.device,
                        probes=count,
                        upload_ms=upload,
                        cpu_ms=cpu,
                        gpu_ms=gpu,
                        speedup=median(cpu) / median(gpu),
                    )
                ),
                flush=True,
            )
