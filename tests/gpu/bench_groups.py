"""Compare ordered merges of independent groups. Upload and verification are outside the timer."""

import argparse
import hashlib
from statistics import median
from time import perf_counter

import numpy as np
from gr_tdigest import TDigest
from gr_tdigest.gpu import GpuDigest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=2500)
    parser.add_argument("--children", type=int, default=40)
    parser.add_argument("--observations", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weighted", action="store_true", help="Use positive fractional weights.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--serial-gpu", action="store_true", help="Also time one unbatched merge_all per group.")
    args = parser.parse_args()
    rng = np.random.default_rng(414)

    def child():
        xs = rng.normal(size=args.observations)
        weights = {"weights": rng.uniform(0.05, 3.0, len(xs))} if args.weighted else {}
        return TDigest.from_array(xs, max_size=args.capacity, **weights)

    cpu = [[child() for _ in range(args.children)] for _ in range(args.groups)]
    gpu = [[GpuDigest(d, args.device) for d in group] for group in cpu]
    expected = [TDigest.merge_all(group).to_bytes() for group in cpu]
    print("SHA-256 of ordered TDIG outputs", hashlib.sha256(b"".join(expected)).hexdigest(), flush=True)
    ops = {
        "CPU loop": lambda: [TDigest.merge_all(group) for group in cpu],
        "CPU groups": lambda: TDigest.merge_groups(cpu),
        "GPU groups": lambda: GpuDigest.merge_groups(gpu, device=args.device),
    }
    if args.serial_gpu:
        ops["GPU loop"] = lambda: [GpuDigest.merge_all(group) for group in gpu]
    times = {name: [] for name in ops}
    order = list(ops)
    for _ in range(args.rounds):
        for name in order:
            start = perf_counter()
            output = ops[name]()
            times[name].append(perf_counter() - start)
            assert [d.to_bytes() for d in output] == expected, name
            del output
        order = order[1:] + order[:1]
    for name, samples in times.items():
        print(f"{name}: median {median(samples):.3f}s, range {min(samples):.3f}–{max(samples):.3f}s")


if __name__ == "__main__":
    main()
