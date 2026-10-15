#!/usr/bin/env python3
"""Compare portable tdigest benchmark result files without running benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(r["kind"], r.get("n"), r["precision"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate case keys in {path}")
    return keyed


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def timing_nodes(value, prefix=""):
    if isinstance(value, dict):
        if "median_ns" in value and "samples_ns" in value:
            yield prefix, value
        else:
            for key, item in value.items():
                yield from timing_nodes(item, f"{prefix}.{key}" if prefix else key)


def inferred_layout(rows, precision):
    points = sorted(
        {
            (r["modern_centroids"], r["modern_retained_bytes_lower_bound"])
            for r in rows.values()
            if r["precision"] == precision and "modern_retained_bytes_lower_bound" in r
        }
    )
    if len(points) < 2:
        return None
    (k0, b0), (k1, b1) = points[0], points[-1]
    width = (b1 - b0) / (k1 - k0)
    obj = b0 - width * k0
    if not all(obj + width * k == size for k, size in points):
        return {"consistent_linear_layout": False, "points": points}
    return {
        "object_bytes": obj,
        "centroid_bytes": width,
        "consistent_linear_layout": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--baseline-metadata", required=True, type=Path)
    p.add_argument("--candidate-metadata", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    before = read_rows(args.baseline)
    after = read_rows(args.candidate)
    if before.keys() != after.keys():
        raise ValueError(
            "Case sets differ; do not compare these as an unchanged matrix"
        )
    bm = json.loads(args.baseline_metadata.read_text())
    cm = json.loads(args.candidate_metadata.read_text())
    for label, meta in [("baseline", bm), ("candidate", cm)]:
        if meta.get("exit_status", meta.get("returncode", 0)) != 0:
            raise ValueError(f"{label} run did not complete successfully")
        if meta.get("source_changed_during_run"):
            raise ValueError(f"{label} source changed during timing")
    for field in ["harness_sha256", "original_pin", "rustc", "environment_overrides"]:
        if bm.get(field) != cm.get(field):
            raise ValueError(f"Unmatched {field}; inspect methodology before comparing")
    changes, times, memory, control_failures = [], [], [], []
    for key in before:
        a, b = before[key], after[key]
        case = {"kind": key[0], "n": key[1], "precision": key[2]}
        at = dict(timing_nodes(a))
        bt = dict(timing_nodes(b))
        if at.keys() != bt.keys():
            raise ValueError(f"Timing metric sets differ for {key}")
        for path in at:
            old, new = at[path]["median_ns"], bt[path]["median_ns"]
            times.append(
                {
                    **case,
                    "operation": path,
                    "baseline_ns": old,
                    "candidate_ns": new,
                    "candidate_over_baseline": new / old if old else None,
                    "baseline_range_ns": [at[path]["min_ns"], at[path]["max_ns"]],
                    "candidate_range_ns": [bt[path]["min_ns"], bt[path]["max_ns"]],
                }
            )
        al, bl = dict(leaves(a)), dict(leaves(b))
        for path in sorted(al.keys() | bl.keys()):
            if any(path.startswith(t + ".") for t in at):
                continue
            old, new = al.get(path), bl.get(path)
            if old != new:
                change = {**case, "field": path, "baseline": old, "candidate": new}
                if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                    change["absolute_change"] = abs(new - old)
                changes.append(change)
                if path.startswith("original_") or path.startswith("merge40.original_"):
                    control_failures.append(change)
        if "modern_retained_bytes_lower_bound" in a:
            old, new = (
                a["modern_retained_bytes_lower_bound"],
                b["modern_retained_bytes_lower_bound"],
            )
            memory.append(
                {
                    **case,
                    "baseline_centroids": a["modern_centroids"],
                    "candidate_centroids": b["modern_centroids"],
                    "baseline_logical_bytes": old,
                    "candidate_logical_bytes": new,
                    "change_bytes": new - old,
                    "change_percent": 100 * (new / old - 1),
                    "baseline_wire_bytes": a["modern_wire_bytes"],
                    "candidate_wire_bytes": b["modern_wire_bytes"],
                }
            )
    result = {
        "scope": "Unchanged matrix; cross-run timings descriptive, not paired or statistically conclusive",
        "baseline_head": bm.get("repository_head", bm.get("head")),
        "candidate_head": cm.get("repository_head", cm.get("head")),
        "baseline_status": bm.get("repository_status"),
        "candidate_status": cm.get("repository_status"),
        "matched_harness_sha256": bm.get("harness_sha256"),
        "cases": len(before),
        "cases_with_non_timing_changes": len(
            {(r["kind"], r["n"], r["precision"]) for r in changes}
        ),
        "original_control_failures": control_failures,
        "layouts": {
            precision: {
                "baseline": inferred_layout(before, precision),
                "candidate": inferred_layout(after, precision),
            }
            for precision in ["f32", "f64"]
        },
        "non_timing_changes": changes,
        "timing_comparisons": times,
        "memory_comparisons": memory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "baseline_head",
                    "candidate_head",
                    "cases",
                    "cases_with_non_timing_changes",
                    "layouts",
                ]
            },
            indent=2,
        )
    )
    print(f"Original control failures: {len(control_failures)}; report: {args.output}")


if __name__ == "__main__":
    main()
