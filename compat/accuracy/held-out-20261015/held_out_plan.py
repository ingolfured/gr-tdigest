#!/usr/bin/env python3
"""Print an accuracy-only case manifest. No compilation or core execution."""

import argparse
import json

DISTRIBUTIONS = [
    "uniform",
    "normal",
    "exponential",
    "lognormal",
    "pareto",
    "ties",
    "spike_tail",
    "separated_modes",
]
TRAINING = [1, 7, 42, 20261014, 104729]
HELD_OUT = [13, 97, 271828, 314159, 999983]


def cases(phase):
    seeds = (
        [TRAINING[0]]
        if phase == "smoke"
        else TRAINING
        if phase == "training"
        else HELD_OUT
    )
    sizes = [32] if phase == "smoke" else [32, 1000, 100000]
    return [
        {
            "phase": phase,
            "seed": seed,
            "kind": kind,
            "n": n,
            "precision": precision,
            "original_delta": 300,
            "modern_delta": 300,
            "merge_order": "both-orders" if n == 100000 else "forward",
            "expected_state_records": 20 if n == 100000 else 11,
        }
        for seed in seeds
        for kind in DISTRIBUTIONS
        for n in sizes
        for precision in ["f64", "f32"]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["smoke", "training", "held-out"], default="smoke"
    )
    args = parser.parse_args()
    rows = cases(args.phase)
    print(
        json.dumps(
            {
                "status": "design only; no datasets generated and no core executed",
                "study": "same-delta accuracy; matched-memory tuning not implemented",
                "datasets": len(rows),
                "state_records": sum(x["expected_state_records"] for x in rows),
                "timing_operations": 0,
                "cases": rows,
            },
            indent=2,
        )
    )
