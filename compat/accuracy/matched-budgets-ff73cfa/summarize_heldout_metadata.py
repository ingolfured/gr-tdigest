"""Compact policy reporting from completed metadata; never open compressed cases."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from summarize_selection import EXPECTED_SHA, PROTOCOL, require, summarize


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def routes(n, core):
    result = {"direct"}
    for layout in ("balanced_iid", "balanced_range", "skewed"):
        for order in ("forward", "reverse") if n == 100000 else ("forward",):
            result.add(f"{layout}/{order}/sequential")
            if core == "modern":
                result.add(f"{layout}/{order}/kway")
    return result


def policy_key(choice):
    return (
        choice["policy_kind"],
        choice["logical_cap_bytes"]
        if choice["policy_kind"] == "storage"
        else choice["rank_threshold_fraction"],
        choice["precision"],
        choice["core"],
    )


def maximum(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "observations": len(values),
        "nonfinite_or_missing": len(values) - len(finite),
        "finite_max": max(finite) if finite else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-complete-confirmed", action="store_true")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            "Prepared only: metadata completion/provenance -> all 24 policy rows -> route comparisons. No study read."
        )
        return
    require(
        args.run_complete_confirmed and args.study and args.output,
        "Explicit completed-run confirmation, study and output required",
    )
    study = args.study.resolve()
    output = args.output.resolve()
    require(
        output != study and study not in output.parents, "Write outside immutable study"
    )
    require(not output.exists(), "Fresh report directory required")
    selection, selection_sha = read(study / "selection-lock.json")
    base = summarize(selection, selection_sha)
    ledger, ledger_sha = read(study / "ledger.json")
    phases = {}
    phase_hashes = {}
    for phase, expected_cases, expected_states in (
        ("smoke", 128, 1408),
        ("training", 1152, 16128),
        ("held-out", 576, 8064),
    ):
        meta, meta_sha = read(study / phase / "metadata.json")
        require(
            meta_sha == ledger["phases"][phase]["metadata_sha256"],
            "Metadata does not match ledger",
        )
        require(
            meta["status"] == "complete"
            and meta["phase"] == phase
            and ledger["phases"][phase]["status"] == "complete",
            "Study phase incomplete",
        )
        require(
            meta["completed_cases"] == expected_cases
            and meta["record_count"] == expected_states,
            "Phase coverage differs",
        )
        require(
            not meta["failures"] and not meta["source_changes"],
            "Phase records failures or changed source",
        )
        require(
            meta["reference_checks"]["status"] == "pass"
            and meta["reference_checks"]["tiny_numeric_cells"] > 0,
            "Reference checks incomplete",
        )
        require(
            meta["protocol"] == selection["protocol"],
            "Protocol differs from frozen selection",
        )
        _, protocol_sha = read(study / phase / "protocol.json")
        require(protocol_sha == PROTOCOL, "Protocol file hash changed")
        phases[phase], phase_hashes[phase] = meta, meta_sha
    training, held = phases["training"], phases["held-out"]
    require(
        selection["training_metadata_sha256"] == phase_hashes["training"],
        "Selection training provenance differs",
    )
    require(
        selection["training_raw_sha256"] == training["compressed_case_sha256"],
        "Selection raw manifest differs",
    )
    retained_selection, retained_sha = read(study / "held-out/selection-lock.json")
    replay, _ = read(study / "held-out/recomputed-selection-lock.json")
    require(
        retained_sha == EXPECTED_SHA == held["selection_lock_sha256"]
        and replay == retained_selection == selection,
        "Held-out settings differ from training lock",
    )
    for name in (
        "source_sha256",
        "binary_sha256",
        "resolved_lock_sha256",
        "build_provenance_sha256",
    ):
        require(
            all(meta[name] == held[name] for meta in phases.values()),
            f"Cross-phase identity differs: {name}",
        )
    choices = {policy_key(row): row for row in base["choices"]}
    expected = set()
    design = selection["design"]
    for kind, thresholds in (
        ("storage", design["logical_storage_caps_bytes"]),
        ("rank", design["illustrative_max_queried_rank_error_budgets"]),
    ):
        for threshold in thresholds:
            for precision in design["mean_precisions"]:
                if all(
                    choices[(kind, threshold, precision, core)]["status"] != "selected"
                    for core in ("original", "modern")
                ):
                    continue
                for seed in design["held_out_seeds"]:
                    for distribution in design["distributions"]:
                        for n in design["sizes"]:
                            expected.add(
                                (kind, threshold, precision, seed, distribution, n)
                            )
    require(
        set(design["held_out_seeds"]).isdisjoint(design["training_seeds"]),
        "Seed leakage",
    )
    require(
        len(held["planned_cases"]) == len(held["case_results"]) == len(expected),
        "Case records incomplete",
    )
    observations = defaultdict(list)
    by_route = defaultdict(list)
    paired = []
    inherited = Counter()
    seen = set()
    for planned, result in zip(
        held["planned_cases"], held["case_results"], strict=True
    ):
        case = result["case"]
        require(
            case == planned and not result["failures"], "Case/result identity failure"
        )
        key = tuple(
            case[name]
            for name in ("policy", "threshold", "precision", "seed", "kind", "n")
        )
        require(
            key in expected and key not in seen, "Unexpected or duplicate fresh case"
        )
        seen.add(key)
        per_case = {}
        for item in result["policy_evaluations"]:
            core, route = item["core"], item["route"]
            require(
                (core, route) not in per_case and route in routes(case["n"], core),
                "Unexpected or duplicate route",
            )
            selected = choices[
                (case["policy"], case["threshold"], case["precision"], core)
            ]
            require(
                selected["status"] == item["status"] == "selected",
                "Unexpected diagnostic/unavailable result in this fixed lock",
            )
            require(
                case[f"{core}_delta"] == selected["selected_delta"],
                "Delta differs from frozen selection",
            )
            require(
                item["policy"] == case["policy"]
                and item["threshold"] == case["threshold"],
                "Policy identity differs",
            )
            require(
                isinstance(item["logical_bytes"], int) and item["logical_bytes"] > 0,
                "Invalid logical bytes",
            )
            if case["policy"] == "storage":
                violation = item["logical_bytes"] > case["threshold"]
                require(
                    item["excess_bytes"]
                    == max(0, item["logical_bytes"] - case["threshold"]),
                    "Cap excess differs",
                )
            else:
                error = item["max_queried_rank_error"]
                require(
                    error is None or math.isfinite(error) and 0 <= error <= 1,
                    "Invalid rank-error fraction",
                )
                violation = error is None or error > case["threshold"]
            require(item["violation"] == violation, "Policy violation differs")
            full = {**case, **item}
            observations[
                (case["policy"], case["threshold"], case["precision"], core)
            ].append(full)
            by_route[
                (case["policy"], case["threshold"], case["precision"], core, route)
            ].append(full)
            per_case[(core, route)] = full
        require(
            set(per_case)
            == {
                (core, route)
                for core in ("original", "modern")
                for route in routes(case["n"], core)
            },
            "Missing state route",
        )
        inherited.update(result["inherited_original_observations"])
        for (core, route), modern in per_case.items():
            if core != "modern":
                continue
            original_route = route.replace("/kway", "/sequential")
            original = per_case[("original", original_route)]
            paired.append(
                {
                    **{
                        name: case[name]
                        for name in (
                            "policy",
                            "threshold",
                            "precision",
                            "seed",
                            "kind",
                            "n",
                        )
                    },
                    "modern_route": route,
                    "original_route": original_route,
                    "comparison": "different_algorithm_kway_vs_sequential"
                    if route.endswith("/kway")
                    else "same_route",
                    "modern_delta": case["modern_delta"],
                    "original_delta": case["original_delta"],
                    "modern_logical_bytes": modern["logical_bytes"],
                    "original_logical_bytes": original["logical_bytes"],
                    "logical_bytes_ratio": modern["logical_bytes"]
                    / original["logical_bytes"],
                    "modern_policy_violation": modern["violation"],
                    "original_policy_violation": original["violation"],
                    "modern_max_queried_rank_error": modern.get(
                        "max_queried_rank_error"
                    ),
                    "original_max_queried_rank_error": original.get(
                        "max_queried_rank_error"
                    ),
                }
            )
    require(
        seen == expected
        and sum(map(len, observations.values())) == held["record_count"],
        "Coverage mismatch",
    )
    final_choices = []
    for key, choice in choices.items():
        rows = observations[key]
        final_choices.append(
            {
                **choice,
                "held_out_status": "complete_metadata_only"
                if rows
                else choice["held_out_status"],
                "held_out_states": len(rows),
                "held_out_policy_violation_states": sum(
                    row["violation"] for row in rows
                )
                if rows
                else None,
                "held_out_cap_violation_states": sum(row["violation"] for row in rows)
                if rows and key[0] == "storage"
                else None,
                "held_out_rank_policy_violation_states": sum(
                    row["violation"] for row in rows
                )
                if rows and key[0] == "rank"
                else None,
                "held_out_max_logical_bytes": max(
                    (row["logical_bytes"] for row in rows), default=None
                ),
                "held_out_max_cap_excess_bytes": max(
                    (row.get("excess_bytes", 0) for row in rows), default=None
                )
                if key[0] == "storage"
                else None,
                "held_out_rank_error_summary": maximum(
                    [row["max_queried_rank_error"] for row in rows]
                )
                if rows and key[0] == "rank"
                else None,
            }
        )
    route_groups = [
        {
            "policy": key[0],
            "threshold": key[1],
            "precision": key[2],
            "core": key[3],
            "route": key[4],
            "states": len(rows),
            "violation_states": sum(row["violation"] for row in rows),
            "max_logical_bytes": max(row["logical_bytes"] for row in rows),
            "rank_error": maximum([row["max_queried_rank_error"] for row in rows])
            if key[0] == "rank"
            else None,
        }
        for key, rows in sorted(by_route.items())
    ]
    report = {
        "schema": "storage-heldout-metadata-summary-v1",
        "status": "completed_policy_metadata_only",
        "selection_lock_sha256": EXPECTED_SHA,
        "ledger_sha256": ledger_sha,
        "phase_metadata_sha256": phase_hashes,
        "protocol_sha256": PROTOCOL,
        "source_sha256": held["source_sha256"],
        "binary_sha256": held["binary_sha256"],
        "resolved_lock_sha256": held["resolved_lock_sha256"],
        "completed_cases": held["completed_cases"],
        "states": held["record_count"],
        "fresh_seeds": design["held_out_seeds"],
        "choices": final_choices,
        "routes": route_groups,
        "inherited_original_observations": dict(inherited),
        "unavailable_policy_cells": held["unavailable_policy_cells"],
        "limits": [
            "No compressed records opened or raw/native hashes recomputed by this metadata-only step; identities are ledger-bound recorded evidence.",
            "Storage-policy quantile/rank/p50/value/CDF errors require the separate raw streaming step and remain pending.",
            "Rank-error values are fractions of empirical probability; multiply by 100 for percentage points. These policy thresholds are not product tolerances.",
            "Logical object/live-payload bytes are not retained capacity, allocator overhead, peak memory, process RSS, or wire size.",
        ],
    }
    require(
        read(study / "ledger.json")[1] == ledger_sha,
        "Study ledger changed during reporting",
    )
    require(
        all(
            read(study / phase / "metadata.json")[1] == digest
            for phase, digest in phase_hashes.items()
        ),
        "Study metadata changed during reporting",
    )
    require(
        read(study / "selection-lock.json")[1] == EXPECTED_SHA,
        "Selection changed during reporting",
    )
    output.mkdir(parents=True)
    (output / "policy-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "route-pairs.jsonl").open("w", encoding="utf-8") as stream:
        for row in paired:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "choices": len(final_choices),
                "states": held["record_count"],
                "same_route_pairs": sum(
                    row["comparison"] == "same_route" for row in paired
                ),
                "kway_vs_sequential_pairs": sum(
                    row["comparison"] != "same_route" for row in paired
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
