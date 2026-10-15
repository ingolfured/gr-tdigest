#!/usr/bin/env python3
"""Lightweight descriptive tables and exact notable records; no measurements."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = [
    json.loads(line) for line in (root / "state-metrics.jsonl").read_text().splitlines()
]
paired = [
    json.loads(line) for line in (root / "paired-drift.jsonl").read_text().splitlines()
]
meta = json.loads((root / "metadata.json").read_text())
print(
    "METADATA",
    json.dumps(
        {
            key: meta.get(key)
            for key in [
                "started_utc",
                "finished_utc",
                "runtime_seconds_execution_metadata",
                "binary_sha256",
                "resolved_lock_sha256",
                "toolchain",
                "protocol",
            ]
        },
        indent=2,
    ),
)
print("DISTRIBUTION MAXIMA: kind,n,core,states,p50_rank,max_rank,cdf,p50_value,p50_iqr")
for kind in sorted({r["kind"] for r in rows}):
    for n in [32, 1000, 100000]:
        for core in ["original", "modern"]:
            subset = [
                r for r in rows if (r["kind"], r["n"], r["core"]) == (kind, n, core)
            ]
            values = []
            for metric in [
                "p50_rank_error",
                "max_rank_error",
                "max_cdf_error",
                "p50_abs_type7_error",
                "p50_error_over_iqr",
            ]:
                valid = [r[metric] for r in subset if r[metric] is not None]
                values.append(max(valid) if valid else None)
            print(json.dumps([kind, n, core, len(subset), *values]))

print(
    "ROUTE MAXIMA n100000: route,core,p50_rank,max_rank,cdf,median_payload,max_payload"
)
for route in sorted({r["route"] for r in rows}):
    for core in ["original", "modern"]:
        subset = [
            r for r in rows if (r["route"], r["core"], r["n"]) == (route, core, 100000)
        ]
        if not subset:
            continue
        values = []
        for metric in ["p50_rank_error", "max_rank_error", "max_cdf_error"]:
            valid = [r[metric] for r in subset if r[metric] is not None]
            values.append(max(valid) if valid else None)
        values += [
            statistics.median(r["logical_retained_bytes"] for r in subset),
            max(r["logical_retained_bytes"] for r in subset),
        ]
        print(json.dumps([route, core, *values]))

print(
    "PAIRED: kind,states,p50_rank_lower/equal/higher,max_rank_lower/equal/higher,median_payload_ratio,max_payload_ratio"
)
for kind in sorted({r["kind"] for r in rows}):
    subset = [r for r in paired if r["kind"] == kind]
    counts = []
    for metric in ["p50_rank_error_change", "max_rank_error_change"]:
        counts.append(
            [
                sum(r[metric] < 0 for r in subset),
                sum(r[metric] == 0 for r in subset),
                sum(r[metric] > 0 for r in subset),
            ]
        )
    print(
        json.dumps(
            [
                kind,
                len(subset),
                *counts,
                statistics.median(r["retained_payload_ratio"] for r in subset),
                max(r["retained_payload_ratio"] for r in subset),
            ]
        )
    )

notable, observations = [], Counter()
by_kind = defaultdict(Counter)
for path in sorted((root / "cases").glob("*.jsonl.gz")):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            qs = row["state"]["quantile"]["rows"]
            if row["core"] == "original":
                nonfinite = sum(q[2] is None for q in qs)
                below = sum(q[2] is not None and q[2] < qs[0][3] for q in qs)
                above = sum(q[2] is not None and q[2] > qs[-1][3] for q in qs)
                observations.update(
                    nonfinite_values=nonfinite,
                    below_min_values=below,
                    above_max_values=above,
                )
                by_kind[row["kind"]].update(
                    nonfinite_states=bool(nonfinite),
                    below_min_states=bool(below),
                    above_max_states=bool(above),
                )
            wanted = (
                row["kind"] == "separated_modes"
                and row["seed"] == 999983
                and row["route"] == "direct"
                and row["n"] in [1000, 100000]
            )
            if wanted:
                entry = {
                    key: row[key]
                    for key in [
                        "kind",
                        "n",
                        "seed",
                        "precision",
                        "route",
                        "core",
                        "input_fnv1a64",
                    ]
                }
                entry["case_file"] = path.name
                entry["p50"] = next(q for q in qs if q[1] == 0.5)
                entry["worst_rank_query"] = max(
                    qs, key=lambda q: q[4] if q[4] is not None else float("inf")
                )
                entry["worst_value_query"] = max(
                    qs, key=lambda q: q[5] if q[5] is not None else float("inf")
                )
                cdf = row["state"].get("cdf", {}).get("rows", [])
                entry["worst_cdf_probe"] = max(cdf, key=lambda q: q[3]) if cdf else None
                entry["state"] = {
                    key: value
                    for key, value in row["state"].items()
                    if key not in ["quantile", "cdf", "drift_vs_original_at_same_query"]
                }
                notable.append(entry)
print(
    "ORIGINAL OBSERVATIONS", json.dumps(dict(observations)), json.dumps(dict(by_kind))
)
(root / "notable-cases.json").write_text(json.dumps(notable, indent=2) + "\n")
print("NOTABLE", json.dumps(notable, indent=2))

print("SAME ROUTE COMPARISONS (kway excluded)")
same_route = [r for r in paired if not r["route"].endswith("/kway")]
for title, subset in [
    ("all", same_route),
    ("n100000", [r for r in same_route if r["n"] == 100000]),
]:
    output = {"subset": title, "states": len(subset)}
    for metric in ["p50_rank_error_change", "max_rank_error_change"]:
        output[metric] = {
            "lower": sum(r[metric] < 0 for r in subset),
            "equal": sum(r[metric] == 0 for r in subset),
            "higher": sum(r[metric] > 0 for r in subset),
            "max_increase": max(r[metric] for r in subset),
        }
    output["max_abs_p50_drift"] = max(abs(r["p50_estimate_drift"]) for r in subset)
    print(json.dumps(output))
for precision in ["f32", "f64"]:
    for core in ["original", "modern"]:
        subset = [
            r
            for r in rows
            if r["n"] == 100000 and r["precision"] == precision and r["core"] == core
        ]
        print(
            json.dumps(
                {
                    "memory_subset": [precision, core],
                    "object_and_payload_min": min(
                        r["logical_retained_bytes"] for r in subset
                    ),
                    "median": statistics.median(
                        r["logical_retained_bytes"] for r in subset
                    ),
                    "max": max(r["logical_retained_bytes"] for r in subset),
                    "centroid_min": min(r["centroid_count"] for r in subset),
                    "centroid_median": statistics.median(
                        r["centroid_count"] for r in subset
                    ),
                    "centroid_max": max(r["centroid_count"] for r in subset),
                }
            )
        )
