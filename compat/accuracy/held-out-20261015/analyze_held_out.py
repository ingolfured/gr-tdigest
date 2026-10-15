#!/usr/bin/env python3
"""Stream completed accuracy records into compact descriptive summaries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(values):
    xs = sorted(value for value in values if value is not None and math.isfinite(value))
    return {
        "n": len(values),
        "finite_n": len(xs),
        "median": statistics.median(xs) if xs else None,
        "p95_nearest_rank": xs[math.ceil(0.95 * len(xs)) - 1] if xs else None,
        "max": xs[-1] if xs else None,
    }


def difference(a, b):
    """Keep a nonfinite result visible as missing, never turn it into zero."""
    return a - b if a is not None and b is not None else None


def worst_key(value):
    return value if value is not None else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["phase"] == "held-out" and metadata["completed_cases"] == 240
    assert metadata["record_count"] == 3360 and metadata["status"] in [
        "complete",
        "failed",
    ]
    assert file_hash(root / "held-out-accuracy") == metadata["binary_sha256"]
    assert file_hash(root / "harness/Cargo.lock") == metadata["resolved_lock_sha256"]
    assert all(
        file_hash(root / name) == expected
        for name, expected in metadata["source_sha256"].items()
    )
    assert json.loads((root / "protocol.json").read_text()) == metadata["protocol"]
    records, original_observations, failures = [], Counter(), []
    compressed_bytes = 0
    for name, expected in metadata["compressed_case_sha256"].items():
        path = root / "cases" / name
        assert file_hash(path) == expected
        compressed_bytes += path.stat().st_size
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                state = row["state"]
                quantiles = state["quantile"]["rows"]
                median = next(q for q in quantiles if q[1] == 0.5)
                ranks = [q[4] for q in quantiles if q[4] is not None]
                values = [q[5] for q in quantiles if q[5] is not None]
                cdf = state.get("cdf", {}).get("rows", [])
                cdf_errors = [q[3] for q in cdf if q[3] is not None]
                item = {
                    key: row[key]
                    for key in ["kind", "n", "seed", "precision", "route", "core"]
                }
                item.update(
                    centroid_count=state["centroids"],
                    logical_retained_bytes=state["logical_retained_bytes"],
                    p50_estimate=median[2],
                    p50_type7_reference=median[3],
                    p50_rank_error=median[4],
                    p50_abs_type7_error=state["quantile"]["p50_abs_error"],
                    p50_error_over_iqr=state["quantile"]["p50_error_over_iqr"],
                    max_rank_error=max(ranks) if ranks else None,
                    max_abs_type7_error=max(values) if values else None,
                    max_cdf_error=max(cdf_errors) if cdf_errors else None,
                    cdf_probe_count=len(cdf),
                    quantile_probe_count=len(quantiles),
                    empty_shards_skipped=row["empty_shards_skipped"],
                    inherited_original_observations=row[
                        "inherited_original_observations"
                    ],
                    invariant_failures=row["invariant_failures"],
                )
                original_observations.update(row["inherited_original_observations"])
                if row["invariant_failures"]:
                    failures.append(
                        {
                            key: item[key]
                            for key in [
                                "kind",
                                "n",
                                "seed",
                                "precision",
                                "route",
                                "invariant_failures",
                            ]
                        }
                    )
                records.append(item)
    assert len(records) == 3360
    metrics = [
        "p50_rank_error",
        "p50_abs_type7_error",
        "p50_error_over_iqr",
        "max_rank_error",
        "max_abs_type7_error",
        "max_cdf_error",
        "centroid_count",
        "logical_retained_bytes",
    ]
    grouped = defaultdict(list)
    for row in records:
        grouped[
            tuple(row[key] for key in ["kind", "n", "precision", "route", "core"])
        ].append(row)
    groups = []
    for key, rows in sorted(grouped.items()):
        assert len(rows) == 5
        groups.append(
            {
                "kind": key[0],
                "n": key[1],
                "precision": key[2],
                "route": key[3],
                "core": key[4],
                "seeds": [r["seed"] for r in rows],
                "metrics": {
                    metric: summary([r[metric] for r in rows]) for metric in metrics
                },
            }
        )
    originals = {
        (r["kind"], r["n"], r["seed"], r["precision"], r["route"]): r
        for r in records
        if r["core"] == "original"
    }
    paired = []
    for modern in [r for r in records if r["core"] == "modern"]:
        original_route = modern["route"].replace("/kway", "/sequential")
        old = originals[
            (
                modern["kind"],
                modern["n"],
                modern["seed"],
                modern["precision"],
                original_route,
            )
        ]
        item = {k: modern[k] for k in ["kind", "n", "seed", "precision", "route"]}
        item.update(
            original_route=original_route,
            p50_estimate_drift=difference(modern["p50_estimate"], old["p50_estimate"]),
            p50_rank_error_change=difference(
                modern["p50_rank_error"], old["p50_rank_error"]
            ),
            max_rank_error_change=difference(
                modern["max_rank_error"], old["max_rank_error"]
            ),
            retained_payload_ratio=modern["logical_retained_bytes"]
            / old["logical_retained_bytes"],
        )
        paired.append(item)
    modern_rows = [r for r in records if r["core"] == "modern"]
    result = {
        "status": metadata["status"],
        "source_protocol": metadata["protocol"],
        "datasets": metadata["completed_cases"],
        "states": len(records),
        "core_states": dict(Counter(r["core"] for r in records)),
        "modern_invariant_failures": failures,
        "driver_failures": metadata["failures"],
        "inherited_original_observations": dict(original_observations),
        "compressed_bytes": compressed_bytes,
        "original_and_modern_primary_quantile_values": sum(
            r["quantile_probe_count"] for r in records
        ),
        "modern_cdf_values_before_roundtrip": sum(
            r["cdf_probe_count"] for r in modern_rows
        ),
        "reference_checks": metadata["reference_checks"],
        "worst_modern_p50_rank": sorted(
            modern_rows, key=lambda r: worst_key(r["p50_rank_error"]), reverse=True
        )[:10],
        "worst_modern_cdf": sorted(
            modern_rows, key=lambda r: worst_key(r["max_cdf_error"]), reverse=True
        )[:10],
        "groups": groups,
        "scope": "Five held-out seeds, fixed delta; descriptive errors against post-cast samples. No timings, matched-memory comparison, universal error bound or global ranking.",
    }
    with (root / "state-metrics.jsonl").open("w") as out:
        for record in records:
            out.write(json.dumps(record, allow_nan=False) + "\n")
    with (root / "paired-drift.jsonl").open("w") as out:
        for record in paired:
            out.write(json.dumps(record, allow_nan=False) + "\n")
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in [
                    "groups",
                    "source_protocol",
                    "worst_modern_p50_rank",
                    "worst_modern_cdf",
                ]
            },
            indent=2,
        )
    )
    for title, subset in [
        ("all modern", modern_rows),
        ("n100000 modern", [r for r in modern_rows if r["n"] == 100000]),
        (
            "n100000 direct",
            [r for r in modern_rows if r["n"] == 100000 and r["route"] == "direct"],
        ),
        (
            "n100000 merged",
            [r for r in modern_rows if r["n"] == 100000 and r["route"] != "direct"],
        ),
    ]:
        print(
            json.dumps(
                {
                    "subset": title,
                    "states": len(subset),
                    "metrics": {
                        metric: summary([r[metric] for r in subset])
                        for metric in [
                            "p50_rank_error",
                            "max_rank_error",
                            "max_cdf_error",
                            "logical_retained_bytes",
                        ]
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
