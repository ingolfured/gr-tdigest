#!/usr/bin/env python3
"""Static storage-cap study plan. This file never invokes a library or build."""

from __future__ import annotations

import argparse
import json

MODERN = "ff73cfab31d6368d203c1a361b8efaaea29d735b"
ORIGINAL = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"
DELTAS = [25, 50, 100, 150, 200, 300, 600, 1200]
BUDGETS = [1024, 2048, 4096]
RANK_BUDGETS = [0.005, 0.01, 0.02]
TRAINING = [1009, 10007, 65537]
HELD_OUT = [15485863, 32452843, 49979687]
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
SIZES = [32, 1000, 100000]
PRECISIONS = ["f64", "f32"]
SCHEMA = "logical-storage-cap-accuracy-v1"


def design():
    return {
        "schema": SCHEMA,
        "modern_commit": MODERN,
        "original_commit": ORIGINAL,
        "delta_grid": DELTAS,
        "logical_storage_caps_bytes": BUDGETS,
        "illustrative_max_queried_rank_error_budgets": RANK_BUDGETS,
        "training_seeds": TRAINING,
        "held_out_seeds": HELD_OUT,
        "distributions": DISTRIBUTIONS,
        "sizes": SIZES,
        "mean_precisions": PRECISIONS,
        "scale": "K2Norm",
        "singleton_policy": "Off",
        "reverse_shard_order_at_n": 100000,
        "selection": "largest delta whose maximum training logical retained bytes fits the cap over all supported routes; one global delta per core/precision/cap",
        "rank_budget_selection": "minimum grid delta whose maximum queried training rank error over all datasets and supported routes is <= threshold; not a product tolerance or universal bound",
        "unreachable": "unavailable within grid; no over-budget substitution; an available single-core cell may run with the unavailable companion at delta25 labelled diagnostic-only",
        "held_out_budget_violation": "retain, flag and report; never retune",
        "comparison": "common logical storage cap, not equal actual storage/capacity/RSS or optimal accuracy",
        "timed_operations": 0,
        "phase_runtime_seconds": {"smoke": 180, "training": 720, "held-out": 600},
        "global_dataset_runtime_seconds": 1200,
        "phase_compressed_output_bytes": {
            "smoke": 128 * 1024**2,
            "training": 1152 * 1024**2,
            "held-out": 768 * 1024**2,
        },
        "global_compressed_output_bytes": 2 * 1024**3,
        "case_timeout_seconds": 120,
        "build_timeout_seconds": 1800,
        "max_json_line_bytes": 8 * 1024**2,
    }


def routes(n, core):
    yield "direct"
    for layout in ["balanced_iid", "balanced_range", "skewed"]:
        for order in ["forward", "reverse"] if n == 100000 else ["forward"]:
            yield f"{layout}/{order}/sequential"
            if core == "modern":
                yield f"{layout}/{order}/kway"


def training_cases(smoke=False):
    for delta in DELTAS:
        for seed in [1] if smoke else TRAINING:
            for kind in DISTRIBUTIONS:
                for n in [32] if smoke else SIZES:
                    for precision in PRECISIONS:
                        yield {
                            "delta": delta,
                            "seed": seed,
                            "kind": kind,
                            "n": n,
                            "precision": precision,
                            "original_delta": delta,
                            "modern_delta": delta,
                            "expected_state_records": 20 if n == 100000 else 11,
                        }


def expected_training_keys():
    return {
        (
            case["delta"],
            case["seed"],
            case["kind"],
            case["n"],
            case["precision"],
            core,
            route,
        )
        for case in training_cases()
        for core in ["original", "modern"]
        for route in routes(case["n"], core)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "training"], default="training")
    args = parser.parse_args()
    cases = list(training_cases(smoke=args.phase == "smoke"))
    print(
        json.dumps(
            {
                "status": "prepared protocol, not executed; fresh full-source gates and grid smoke required",
                "phase": args.phase,
                "invocations": len(cases),
                "states": sum(case["expected_state_records"] for case in cases),
                "design": design(),
            },
            indent=2,
        )
    )
