"""Bounded, one-pass reduction of completed compressed held-out evidence."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import time

from summarize_heldout_metadata import routes
from summarize_selection import EXPECTED_SHA, require


METADATA_REPORT_SHA = "14a827d8718f65cdc50985903ec0daeb0ac35b5690daf75ea30499e5deee1c9f"
HELPERS = {
    "summarize_heldout_metadata.py": "e43754619f7da80688e5be06b363fc4619e63506922d24bf4bdcda5d5092797a",
    "summarize_selection.py": "c6aaf4c9c3cac8600e0bf2e9285bfbf0e74aad30167c86f29b74b21788be4b29",
}
MAX_SECONDS = 600
MAX_DECOMPRESSED = 8 * 1024**3
MAX_OUTPUT = 64 * 1024**2
MAX_LINE = 8 * 1024**2
TAILS = (0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999)
QUERIES = sorted(set([i / 1000 for i in range(1001)] + [0.0001, 0.9999]))


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.decompressed = 0
        self.output = 0

    def check(self):
        if time.monotonic() - self.started > MAX_SECONDS:
            raise BudgetStop("600-second processing budget reached")
        if self.decompressed > MAX_DECOMPRESSED:
            raise BudgetStop("8 GiB decompressed-byte budget reached")
        if self.output > MAX_OUTPUT:
            raise BudgetStop("64 MiB output budget reached")


def file_hash(path, budget):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            budget.check()
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


class HashedReader(io.RawIOBase):
    """Hash compressed bytes as gzip consumes them, without a second raw pass."""

    def __init__(self, source, budget):
        self.source = source
        self.budget = budget
        self.hash = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        self.budget.check()
        count = self.source.readinto(buffer)
        if count:
            self.hash.update(memoryview(buffer)[:count])
            self.count += count
        return count


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def all_max(values):
    # A missing/nonfinite observation cannot disappear from the maximum.
    return max(values) if values and all(finite(v) for v in values) else None


def stats(values):
    good = sorted(v for v in values if finite(v))
    missing = len(values) - len(good)
    return {
        "observations": len(values),
        "nonfinite_or_missing": missing,
        "max": good[-1] if good and not missing else None,
        "finite_max": good[-1] if good else None,
        "finite_median": statistics.median(good) if good else None,
        "finite_min": good[0] if good else None,
    }


def delta(a, b):
    return a - b if finite(a) and finite(b) else None


def qpoint(row):
    return dict(
        zip(
            (
                "requested_q",
                "effective_q",
                "estimate",
                "type7_reference",
                "rank_error",
                "abs_value_error",
            ),
            row,
            strict=True,
        )
    )


def worst(rows, index):
    return max(rows, key=lambda r: r[index] if finite(r[index]) else float("inf"))


def compact(row, case, filename, compressed_sha, number):
    state = row["state"]
    core = row["core"]
    require(
        core in ("original", "modern") and row["route"] in routes(case["n"], core),
        "Unexpected raw route",
    )
    require(
        all(
            row[name] == case[name]
            for name in (
                "kind",
                "n",
                "seed",
                "precision",
                "original_delta",
                "modern_delta",
            )
        ),
        "Raw case identity differs",
    )
    require(not row["invariant_failures"], "Recorded modern invariant failure")
    require(state["count"] == case["n"], "Recorded count differs")
    require(
        state["logical_retained_bytes"]
        == state["object_bytes"] + state["centroid_bytes"] * state["centroids"],
        "Logical byte formula differs",
    )
    require(
        row["shard_sizes"] == []
        if row["route"] == "direct"
        else len(row["shard_sizes"]) == 40 and sum(row["shard_sizes"]) == case["n"],
        "Shard coverage differs",
    )
    quantile = state["quantile"]
    require(
        quantile["columns"]
        == [
            "requested_q",
            "effective_q",
            "estimate",
            "type7_reference",
            "rank_error",
            "abs_value_error",
        ],
        "Quantile columns differ",
    )
    qs = quantile["rows"]
    require(len(qs) == 1003 and [q[0] for q in qs] == QUERIES, "Quantile grid differs")
    for q in qs:
        require(len(q) == 6 and finite(q[3]), "Invalid reference row")
        effective = (
            struct.unpack("f", struct.pack("f", q[0]))[0]
            if case["precision"] == "f32"
            else q[0]
        )
        require(q[1] == effective, "Effective query rounding differs")
        require(q[4] is None or finite(q[4]) and 0 <= q[4] <= 1, "Invalid rank error")
        require(q[5] is None or finite(q[5]) and q[5] >= 0, "Invalid value error")
        require(
            q[5] == abs(q[2] - q[3]) if finite(q[2]) else q[5] is None,
            "Value error differs from estimate/reference",
        )
    p50 = next(q for q in qs if q[0] == 0.5)
    require(p50[5] == quantile["p50_abs_error"], "p50 stored error differs")
    ranks, values = [q[4] for q in qs], [q[5] for q in qs]
    cdf = state.get("cdf", {}).get("rows", [])
    if core == "modern":
        require(state["wire_queries_bit_equal"] is True, "Wire parity not recorded")
        require(
            state["cdf"]["columns"] == ["x", "estimate", "midpoint_ecdf", "abs_error"]
            and cdf,
            "CDF schema differs",
        )
        for c in cdf:
            require(
                len(c) == 4
                and all(finite(v) for v in c)
                and 0 <= c[1] <= 1
                and 0 <= c[2] <= 1,
                "Invalid CDF row",
            )
            require(c[3] == abs(c[1] - c[2]), "CDF error arithmetic differs")
    else:
        require(not cdf, "Original CDF must be absent")
    metric = {
        **{
            name: case[name]
            for name in (
                "policy",
                "threshold",
                "kind",
                "n",
                "seed",
                "precision",
                "original_delta",
                "modern_delta",
            )
        },
        "core": core,
        "route": row["route"],
        "input_fnv1a64": row["input_fnv1a64"],
        "source_case_file": filename,
        "source_case_sha256": compressed_sha,
        "source_record_1based": number,
        "centroids": state["centroids"],
        "logical_bytes": state["logical_retained_bytes"],
        "object_bytes": state["object_bytes"],
        "centroid_bytes": state["centroid_bytes"],
        "vector_capacity_bytes": state["vector_capacity_bytes"],
        "capacity_measurement": "original clone compacted with shrink_to_fit"
        if core == "original"
        else "unknown",
        "wire_bytes": state.get("wire_bytes"),
        "p50": qpoint(p50),
        "p50_rank_error": p50[4],
        "p50_abs_type7_error": p50[5],
        "p50_error_over_iqr": quantile["p50_error_over_iqr"],
        "sample_iqr": quantile["sample_iqr"],
        "max_rank_error": all_max(ranks),
        "max_abs_type7_error": all_max(values),
        "nonfinite_rank_queries": sum(not finite(v) for v in ranks),
        "nonfinite_value_queries": sum(not finite(v) for v in values),
        "worst_rank_query": qpoint(worst(qs, 4)),
        "worst_value_query": qpoint(worst(qs, 5)),
        "tails": [
            qpoint(next(q for q in qs if q[0] == requested)) for requested in TAILS
        ],
        "quantile_probe_count": len(qs),
        "cdf_probe_count": len(cdf),
        "max_cdf_error": all_max([c[3] for c in cdf]) if cdf else None,
        "worst_cdf_probe": dict(
            zip(
                ("x", "estimate", "midpoint_ecdf", "abs_error"),
                worst(cdf, 3),
                strict=True,
            )
        )
        if cdf
        else None,
        "inherited_original_observations": row["inherited_original_observations"],
    }
    if case["policy"] == "storage":
        metric["policy_violation"] = metric["logical_bytes"] > case["threshold"]
        metric["cap_excess_bytes"] = max(0, metric["logical_bytes"] - case["threshold"])
    else:
        metric["policy_violation"] = (
            metric["max_rank_error"] is None
            or metric["max_rank_error"] > case["threshold"]
        )
        metric["rank_excess_fraction"] = (
            max(0, metric["max_rank_error"] - case["threshold"])
            if metric["max_rank_error"] is not None
            else None
        )
    return metric


def group_metrics(rows, include_values):
    names = ["p50_rank_error", "max_rank_error", "logical_bytes", "centroids"]
    if include_values:
        names += ["p50_abs_type7_error", "p50_error_over_iqr", "max_abs_type7_error"]
    if rows[0]["core"] == "modern":
        names += ["max_cdf_error", "wire_bytes"]
    return {name: stats([r[name] for r in rows]) for name in names}


def write_record(stream, value, budget):
    encoded = (
        json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()
    budget.output += len(encoded)
    budget.check()
    stream.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-slot-released", action="store_true")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--metadata-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            "Prepared only; requires independent review and CPU allocation. No files read."
        )
        return
    require(
        args.cpu_slot_released and args.study and args.output and args.metadata_report,
        "Explicit allocation and all paths required",
    )
    study, output = args.study.resolve(), args.output.resolve()
    require(
        not output.exists() and output != study and study not in output.parents,
        "Fresh output outside immutable study required",
    )
    budget = Budget()
    output.mkdir(parents=True)
    status = {
        "status": "preparing",
        "completed_cases": 0,
        "states": 0,
        "failures": [],
        "limits": {
            "seconds": MAX_SECONDS,
            "decompressed_bytes": MAX_DECOMPRESSED,
            "output_bytes": MAX_OUTPUT,
            "line_bytes": MAX_LINE,
        },
    }
    try:
        for name, digest in HELPERS.items():
            require(
                file_hash(Path(__file__).parent / name, budget) == digest,
                "Reviewed helper changed",
            )
        report, report_sha = read_json(args.metadata_report)
        require(
            report_sha == METADATA_REPORT_SHA
            and report["status"] == "completed_policy_metadata_only",
            "Metadata summary identity differs",
        )
        ledger, ledger_sha = read_json(study / "ledger.json")
        require(
            ledger_sha == report["ledger_sha256"],
            "Ledger changed after metadata report",
        )
        meta, meta_sha = read_json(study / "held-out/metadata.json")
        require(
            meta_sha
            == report["phase_metadata_sha256"]["held-out"]
            == ledger["phases"]["held-out"]["metadata_sha256"],
            "Held-out metadata changed",
        )
        require(
            meta["status"] == "complete"
            and meta["completed_cases"] == 576
            and meta["record_count"] == 8064
            and not meta["failures"]
            and not meta["source_changes"],
            "Held-out phase incomplete",
        )
        selection, selection_sha = read_json(study / "selection-lock.json")
        require(
            selection_sha == EXPECTED_SHA == meta["selection_lock_sha256"],
            "Frozen selection changed",
        )
        phase = study / "held-out"
        require(
            file_hash(phase / "held-out-accuracy", budget) == meta["binary_sha256"],
            "Native binary changed",
        )
        require(
            file_hash(phase / "harness/Cargo.lock", budget)
            == meta["resolved_lock_sha256"],
            "Resolved lock changed",
        )
        for name, digest in meta["source_sha256"].items():
            require(
                file_hash(phase / name, budget) == digest, f"Source changed: {name}"
            )
        require(
            {p.name for p in (phase / "cases").iterdir() if p.is_file()}
            == set(meta["compressed_case_sha256"]),
            "Compressed case coverage differs",
        )
        status.update(
            status="running",
            metadata_report_sha256=report_sha,
            ledger_sha256=ledger_sha,
            metadata_sha256=meta_sha,
            selection_lock_sha256=selection_sha,
            source_sha256=meta["source_sha256"],
            binary_sha256=meta["binary_sha256"],
            resolved_lock_sha256=meta["resolved_lock_sha256"],
            reducer_sha256=file_hash(Path(__file__), budget),
            helper_sha256=HELPERS,
        )
        records, inherited, raw_hashes = [], Counter(), {}
        same_pairs = different_pairs = 0
        with (
            (output / "state-metrics.jsonl").open("wb") as metrics_out,
            (output / "route-pairs.jsonl").open("wb") as pairs_out,
        ):
            for index, result in enumerate(meta["case_results"]):
                budget.check()
                case = result["case"]
                require(case == meta["planned_cases"][index], "Planned case differs")
                name = f"{index:04d}-{case['policy']}-{case['threshold']}-d{case['original_delta']}-{case['modern_delta']}-{case['kind']}-{case['n']}-{case['precision']}-{case['seed']}.jsonl.gz"
                expected_sha = meta["compressed_case_sha256"][name]
                require(result["sha256"] == expected_sha, "Result/file digest differs")
                per_case, raw_case = {}, {}
                with (phase / "cases" / name).open("rb") as compressed:
                    reader = HashedReader(compressed, budget)
                    with (
                        io.BufferedReader(reader) as buffered,
                        gzip.GzipFile(fileobj=buffered) as stream,
                    ):
                        number = 0
                        while line := stream.readline(MAX_LINE + 1):
                            number += 1
                            require(
                                len(line) <= MAX_LINE and line.endswith(b"\n"),
                                "Oversized or incomplete raw line",
                            )
                            budget.decompressed += len(line)
                            budget.check()
                            require(
                                number <= case["expected_state_records"],
                                "Unexpected extra state",
                            )
                            raw = json.loads(line)
                            item = compact(raw, case, name, expected_sha, number)
                            key = (item["core"], item["route"])
                            require(key not in per_case, "Duplicate raw route")
                            per_case[key], raw_case[key] = item, raw
                    require(
                        reader.hash.hexdigest() == expected_sha
                        and reader.count == (phase / "cases" / name).stat().st_size,
                        "Compressed bytes changed",
                    )
                require(
                    set(per_case)
                    == {
                        (core, route)
                        for core in ("original", "modern")
                        for route in routes(case["n"], core)
                    },
                    "Incomplete raw route coverage",
                )
                require(
                    len({item["input_fnv1a64"] for item in per_case.values()}) == 1,
                    "Input fingerprints differ",
                )
                require(
                    number == result["records"] == case["expected_state_records"],
                    "State count differs",
                )
                policy_rows = {
                    (r["core"], r["route"]): r for r in result["policy_evaluations"]
                }
                case_observations = Counter()
                for key, item in per_case.items():
                    recorded = policy_rows[key]
                    require(
                        recorded["logical_bytes"] == item["logical_bytes"]
                        and recorded["violation"] == item["policy_violation"],
                        "Raw policy/storage result differs from metadata",
                    )
                    if case["policy"] == "rank":
                        require(
                            recorded["max_queried_rank_error"]
                            == item["max_rank_error"],
                            "Raw rank result differs from metadata",
                        )
                    case_observations.update(item["inherited_original_observations"])
                    records.append(item)
                    write_record(metrics_out, item, budget)
                    if key[0] == "modern":
                        old_key = ("original", key[1].replace("/kway", "/sequential"))
                        old = per_case[old_key]
                        modern_q = raw_case[key]["state"]["quantile"]["rows"]
                        old_q = raw_case[old_key]["state"]["quantile"]["rows"]
                        require(
                            all(
                                (a[0], a[1], a[3]) == (b[0], b[1], b[3])
                                for a, b in zip(modern_q, old_q, strict=True)
                            ),
                            "Paired queries/reference differ",
                        )
                        drifts = [
                            delta(a[2], b[2])
                            for a, b in zip(modern_q, old_q, strict=True)
                        ]
                        require(
                            drifts
                            == raw_case[key]["state"][
                                "drift_vs_original_at_same_query"
                            ],
                            "Recorded drift differs",
                        )
                        different = key[1].endswith("/kway")
                        same_pairs += not different
                        different_pairs += different
                        pair = {
                            **{
                                k: case[k]
                                for k in (
                                    "policy",
                                    "threshold",
                                    "kind",
                                    "n",
                                    "seed",
                                    "precision",
                                    "original_delta",
                                    "modern_delta",
                                )
                            },
                            "modern_route": key[1],
                            "original_route": old_key[1],
                            "comparison": "different_algorithm_kway_vs_sequential"
                            if different
                            else "same_route",
                            "input_fnv1a64": item["input_fnv1a64"],
                            "p50_estimate_drift": delta(
                                item["p50"]["estimate"], old["p50"]["estimate"]
                            ),
                            "max_abs_estimate_drift": all_max(
                                [abs(d) if d is not None else None for d in drifts]
                            ),
                            "p50_rank_error_change": delta(
                                item["p50_rank_error"], old["p50_rank_error"]
                            ),
                            "max_rank_error_change": delta(
                                item["max_rank_error"], old["max_rank_error"]
                            ),
                            "p50_abs_value_error_change": delta(
                                item["p50_abs_type7_error"], old["p50_abs_type7_error"]
                            ),
                            "logical_bytes_ratio": item["logical_bytes"]
                            / old["logical_bytes"],
                            "modern_policy_violation": item["policy_violation"],
                            "original_policy_violation": old["policy_violation"],
                        }
                        write_record(pairs_out, pair, budget)
                require(
                    dict(case_observations)
                    == result["inherited_original_observations"],
                    "Inherited observations differ",
                )
                inherited.update(case_observations)
                raw_hashes[name] = expected_sha
                status.update(completed_cases=index + 1, states=len(records))
                if (index + 1) % 48 == 0:
                    print(
                        json.dumps({"cases": index + 1, "states": len(records)}),
                        flush=True,
                    )
        require(
            len(records) == 8064
            and same_pairs == 2880
            and different_pairs == 2304
            and raw_hashes == meta["compressed_case_sha256"],
            "Final coverage differs",
        )
        require(
            dict(inherited) == report["inherited_original_observations"],
            "Metadata/record observation totals differ",
        )
        grouped, cells = defaultdict(list), defaultdict(list)
        for row in records:
            grouped[
                tuple(
                    row[k]
                    for k in (
                        "policy",
                        "threshold",
                        "precision",
                        "core",
                        "kind",
                        "n",
                        "route",
                    )
                )
            ].append(row)
            cells[
                tuple(row[k] for k in ("policy", "threshold", "precision", "core"))
            ].append(row)
        groups = []
        for key, rows in sorted(grouped.items()):
            require(
                sorted(row["seed"] for row in rows) == sorted(report["fresh_seeds"]),
                "Incomplete fresh seeds in group",
            )
            groups.append(
                {
                    "identity": dict(
                        zip(
                            (
                                "policy",
                                "threshold",
                                "precision",
                                "core",
                                "kind",
                                "n",
                                "route",
                            ),
                            key,
                            strict=True,
                        )
                    ),
                    "seeds": [row["seed"] for row in rows],
                    "metrics": group_metrics(rows, True),
                    "policy_violation_states": sum(
                        row["policy_violation"] for row in rows
                    ),
                }
            )
        choices = []
        for selected in report["choices"]:
            kind = selected["policy_kind"]
            threshold = (
                selected["logical_cap_bytes"]
                if kind == "storage"
                else selected["rank_threshold_fraction"]
            )
            rows = cells[(kind, threshold, selected["precision"], selected["core"])]
            require(len(rows) == selected["held_out_states"], "Per-cell count differs")
            require(
                sum(row["policy_violation"] for row in rows)
                == (selected["held_out_policy_violation_states"] or 0),
                "Per-cell violation count differs",
            )
            choices.append(
                {
                    **selected,
                    "full_metric_status": "complete"
                    if rows
                    else "unavailable_within_grid",
                    "full_metrics": group_metrics(rows, False) if rows else None,
                }
            )
        final = {
            "status": "complete",
            "choices": choices,
            "groups": groups,
            "states": len(records),
            "datasets_policy_invocations": 576,
            "same_route_pairs": same_pairs,
            "different_algorithm_pairs": different_pairs,
            "rank_error_unit": "probability fraction; multiply by 100 for percentage points",
            "value_error_unit": "absolute input units against Type7 post-cast sample reference, grouped by distribution",
            "cdf_reference": "modern only, midpoint empirical CDF",
            "inherited_original_observations": dict(inherited),
            "quantile_probes": sum(r["quantile_probe_count"] for r in records),
            "modern_cdf_probes": sum(r["cdf_probe_count"] for r in records),
            "scope": "Three fresh seeds; frozen training-selected deltas; descriptive observations, not independent-query significance, universal error bounds, equal RSS, or world ranking.",
        }
        require(
            read_json(study / "ledger.json")[1] == ledger_sha
            and read_json(study / "held-out/metadata.json")[1] == meta_sha
            and read_json(study / "selection-lock.json")[1] == EXPECTED_SHA,
            "Evidence changed during reduction",
        )
        require(
            file_hash(phase / "held-out-accuracy", budget) == meta["binary_sha256"],
            "Native binary changed during reduction",
        )
        require(
            file_hash(phase / "harness/Cargo.lock", budget)
            == meta["resolved_lock_sha256"],
            "Resolved lock changed during reduction",
        )
        for name, digest in meta["source_sha256"].items():
            require(
                file_hash(phase / name, budget) == digest,
                f"Source changed during reduction: {name}",
            )
        for name, digest in HELPERS.items():
            require(
                file_hash(Path(__file__).parent / name, budget) == digest,
                "Reviewed helper changed during reduction",
            )
        require(
            file_hash(Path(__file__), budget) == status["reducer_sha256"],
            "Reducer changed during reduction",
        )
        with (output / "summary.json").open("wb") as stream:
            write_record(stream, final, budget)
        with (output / "raw-file-sha256.json").open("wb") as stream:
            write_record(stream, raw_hashes, budget)
        status["output_sha256"] = {
            name: file_hash(output / name, budget)
            for name in (
                "state-metrics.jsonl",
                "route-pairs.jsonl",
                "summary.json",
                "raw-file-sha256.json",
            )
        }
        status["status"] = "complete"
    except BudgetStop as error:
        status.update(status="incomplete_budget", stopped_reason=str(error))
    except Exception as error:
        status.update(status="failed", failures=[repr(error)])
    finally:
        status.update(
            elapsed_seconds=time.monotonic() - budget.started,
            decompressed_bytes=budget.decompressed,
            generated_bytes=budget.output,
        )
        (output / "status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                k: status[k]
                for k in (
                    "status",
                    "completed_cases",
                    "states",
                    "failures",
                    "elapsed_seconds",
                )
            }
        )
    )
    raise SystemExit(0 if status["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
