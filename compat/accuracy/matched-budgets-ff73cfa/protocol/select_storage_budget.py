#!/usr/bin/env python3
"""Draft selector for a future completed training run; never invokes the core."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from storage_budget_plan import (
    BUDGETS,
    DELTAS,
    PRECISIONS,
    RANK_BUDGETS,
    design,
    expected_training_keys,
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = args.training.resolve()
    if args.output.exists():
        parser.error("Selection lock already exists; use a fresh output path")
    metadata = json.loads((training / "metadata.json").read_text())
    protocol = json.loads((training / "protocol.json").read_text())
    policy = json.loads((training / "policy-lock.json").read_text())
    if metadata["status"] != "complete" or metadata["phase"] != "training":
        raise ValueError("Only complete training evidence can select parameters")
    if metadata["protocol"] != protocol or protocol["design"] != design():
        raise ValueError("Training protocol does not match the predeclared design")
    if (
        policy["design"] != design()
        or digest(training / "policy-lock.json") != protocol["policy_lock_sha256"]
    ):
        raise ValueError("Predeclared policy lock changed")
    for name in ["storage_budget_plan.py", "select_storage_budget.py"]:
        if (
            digest(Path(__file__).with_name(name))
            != policy["source_files_sha256"][name]
        ):
            raise ValueError(f"Selection policy implementation changed: {name}")
    if metadata["failures"] or metadata["source_changes"]:
        raise ValueError("Training has failures/source changes")
    if metadata["completed_cases"] != 1152 or metadata["record_count"] != 16128:
        raise ValueError("Incomplete training coverage")
    if metadata["reference_checks"]["status"] != "pass":
        raise ValueError("Reference validation did not pass")
    # The future runner must retain these artifacts just as the validated runner does.
    if digest(training / "held-out-accuracy") != metadata["binary_sha256"]:
        raise ValueError("Executable changed")
    if digest(training / "harness/Cargo.lock") != metadata["resolved_lock_sha256"]:
        raise ValueError("Resolved dependency lock changed")
    for name, expected in metadata["source_sha256"].items():
        if digest(training / name) != expected:
            raise ValueError(f"Source changed: {name}")
    grouped = defaultdict(list)
    seen = set()
    expected = expected_training_keys()
    for name, expected_hash in metadata["compressed_case_sha256"].items():
        if Path(name).name != name or not name.endswith(".jsonl.gz"):
            raise ValueError("Invalid compressed case filename")
        path = training / "cases" / name
        if digest(path) != expected_hash:
            raise ValueError(f"Compressed data changed: {name}")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                old_delta, modern_delta = row["original_delta"], row["modern_delta"]
                if old_delta != modern_delta or old_delta not in DELTAS:
                    raise ValueError("Unexpected training delta pair")
                key = (
                    old_delta,
                    row["seed"],
                    row["kind"],
                    row["n"],
                    row["precision"],
                    row["core"],
                    row["route"],
                )
                if key not in expected or key in seen:
                    raise ValueError(f"Unexpected/duplicate training state: {key}")
                seen.add(key)
                if row["core"] == "modern" and row["invariant_failures"]:
                    raise ValueError("Modern invariant failure in training raw data")
                state = row["state"]
                size = state["logical_retained_bytes"]
                if not isinstance(size, int) or size <= 0:
                    raise ValueError("Invalid logical storage value")
                if (
                    size
                    != state["object_bytes"]
                    + state["centroids"] * state["centroid_bytes"]
                ):
                    raise ValueError("Logical storage formula mismatch")
                quantiles = state["quantile"]["rows"]
                if len(quantiles) != 1003:
                    raise ValueError("Unexpected training query count")
                ranks = [query[4] for query in quantiles]
                finite_rank = all(
                    value is not None and math.isfinite(value) for value in ranks
                )
                if finite_rank and any(value < 0 or value > 1 for value in ranks):
                    raise ValueError("Invalid rank error")
                max_rank = max(ranks) if finite_rank else None
                grouped[(row["core"], row["precision"], old_delta)].append(
                    {"bytes": size, "max_rank": max_rank}
                )
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} declared training states")
    selections = []
    rank_selections = []
    for core in ["original", "modern"]:
        for precision in PRECISIONS:
            for cap in BUDGETS:
                candidates = []
                for delta in DELTAS:
                    measurements = grouped[(core, precision, delta)]
                    sizes = [item["bytes"] for item in measurements]
                    maximum = max(sizes)
                    candidates.append(
                        {
                            "delta": delta,
                            "states": len(sizes),
                            "max_logical_bytes": maximum,
                            "median_logical_bytes": statistics.median(sizes),
                            "fits_training_cap": maximum <= cap,
                        }
                    )
                feasible = [item for item in candidates if item["fits_training_cap"]]
                chosen = (
                    max(feasible, key=lambda item: item["delta"]) if feasible else None
                )
                selections.append(
                    {
                        "core": core,
                        "precision": precision,
                        "logical_cap_bytes": cap,
                        "status": "selected" if chosen else "unavailable_within_grid",
                        "delta": chosen["delta"] if chosen else None,
                        "max_training_logical_bytes": chosen["max_logical_bytes"]
                        if chosen
                        else None,
                        "training_cap_slack_bytes": cap - chosen["max_logical_bytes"]
                        if chosen
                        else None,
                        "at_grid_upper_edge": chosen["delta"] == max(DELTAS)
                        if chosen
                        else False,
                        "candidates": candidates,
                    }
                )
            for threshold in RANK_BUDGETS:
                candidates = []
                for delta in DELTAS:
                    measurements = grouped[(core, precision, delta)]
                    ranks = [item["max_rank"] for item in measurements]
                    maximum = (
                        max(ranks)
                        if all(value is not None for value in ranks)
                        else None
                    )
                    candidates.append(
                        {
                            "delta": delta,
                            "states": len(measurements),
                            "max_queried_training_rank_error": maximum,
                            "nonfinite_rank_states": sum(
                                value is None for value in ranks
                            ),
                            "passes_training_threshold": maximum is not None
                            and maximum <= threshold,
                            "max_logical_bytes": max(
                                item["bytes"] for item in measurements
                            ),
                        }
                    )
                feasible = [
                    item for item in candidates if item["passes_training_threshold"]
                ]
                chosen = (
                    min(feasible, key=lambda item: item["delta"]) if feasible else None
                )
                rank_selections.append(
                    {
                        "core": core,
                        "precision": precision,
                        "rank_error_threshold": threshold,
                        "threshold_is_product_tolerance": False,
                        "status": "selected" if chosen else "unavailable_within_grid",
                        "delta": chosen["delta"] if chosen else None,
                        "max_queried_training_rank_error": chosen[
                            "max_queried_training_rank_error"
                        ]
                        if chosen
                        else None,
                        "max_training_logical_bytes": chosen["max_logical_bytes"]
                        if chosen
                        else None,
                        "at_grid_lower_edge": chosen["delta"] == min(DELTAS)
                        if chosen
                        else False,
                        "candidates": candidates,
                    }
                )
    lock = {
        "schema": "logical-storage-selection-lock-v1",
        "status": "locked_from_training_only",
        "design": design(),
        "protocol": protocol,
        "protocol_sha256": digest(training / "protocol.json"),
        "training_metadata_sha256": digest(training / "metadata.json"),
        "training_raw_sha256": metadata["compressed_case_sha256"],
        "selector_sha256": digest(Path(__file__)),
        "storage_selections": selections,
        "illustrative_rank_error_selections": rank_selections,
        "policy": "Never adapt these selections to held-out errors, threshold failures or cap violations; unavailable cells remain unavailable",
    }
    with args.output.open("x") as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "storage_cells": len(selections),
                "rank_threshold_cells": len(rank_selections),
                "unavailable_storage": sum(
                    item["delta"] is None for item in selections
                ),
                "unavailable_rank_threshold": sum(
                    item["delta"] is None for item in rank_selections
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
