"""Summarize a frozen training lock only; never load native code or raw samples."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


EXPECTED_SHA = "300203022d0dfbaee3e0a5bcf947bee6f804687b927238cd2e1b0576f16bb562"
MODERN = "ff73cfab31d6368d203c1a361b8efaaea29d735b"
ORIGINAL = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"
PROTOCOL = "9c2a455e689802eb2dbbe7764764238b794fa5975e2517d5e3c1355fed66904a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(data, digest):
    require(digest == EXPECTED_SHA, "Unexpected selection-lock hash")
    require(data["status"] == "locked_from_training_only", "Lock is not frozen")
    design = data["design"]
    require(design["modern_commit"] == MODERN, "Unexpected modern source")
    require(design["original_commit"] == ORIGINAL, "Unexpected original source")
    require(data["protocol_sha256"] == PROTOCOL, "Unexpected execution protocol")
    grid = design["delta_grid"]
    require(grid == [25, 50, 100, 150, 200, 300, 600, 1200], "Unexpected grid")
    keys = [
        (core, precision)
        for core in ("original", "modern")
        for precision in ("f64", "f32")
    ]
    ranks = {}
    storage = {}
    rows = []
    grid_rows = {}
    for kind, field, target, policies in (
        ("storage", "logical_cap_bytes", storage, [1024, 2048, 4096]),
        ("rank", "rank_error_threshold", ranks, [0.005, 0.01, 0.02]),
    ):
        source = data[
            "storage_selections"
            if kind == "storage"
            else "illustrative_rank_error_selections"
        ]
        require(len(source) == 12, "Expected twelve policies of each kind")
        for selection in source:
            key = (selection["core"], selection["precision"], selection[field])
            require(key not in target, "Duplicate policy")
            target[key] = selection
            candidates = selection["candidates"]
            require(
                [c["delta"] for c in candidates] == grid, "Missing/reordered candidate"
            )
            if kind == "storage":
                eligible = [
                    c for c in candidates if c["max_logical_bytes"] <= selection[field]
                ]
                require(
                    all(c["fits_training_cap"] == (c in eligible) for c in candidates),
                    "Cap flags differ",
                )
                winner = eligible[-1] if eligible else None
            else:
                eligible = [
                    c
                    for c in candidates
                    if c["nonfinite_rank_states"] == 0
                    and math.isfinite(c["max_queried_training_rank_error"])
                    and c["max_queried_training_rank_error"] <= selection[field]
                ]
                require(
                    all(
                        c["passes_training_threshold"] == (c in eligible)
                        for c in candidates
                    ),
                    "Rank flags differ",
                )
                winner = eligible[0] if eligible else None
            require(
                selection["delta"] == (winner["delta"] if winner else None),
                "Selection differs from policy",
            )
            require(
                selection["status"]
                == ("selected" if winner else "unavailable_within_grid"),
                "Incorrect availability",
            )
            require(
                selection["max_training_logical_bytes"]
                == (winner["max_logical_bytes"] if winner else None),
                "Selected storage differs",
            )
            if kind == "storage" and winner:
                require(
                    selection["training_cap_slack_bytes"]
                    == selection[field] - winner["max_logical_bytes"],
                    "Incorrect slack",
                )
            if kind == "rank":
                require(
                    selection["max_queried_training_rank_error"]
                    == (winner["max_queried_training_rank_error"] if winner else None),
                    "Selected error differs",
                )
                for c in candidates:
                    candidate_key = (key[0], key[1], c["delta"])
                    observation = {
                        "core": key[0],
                        "precision": key[1],
                        **{
                            k: v
                            for k, v in c.items()
                            if k != "passes_training_threshold"
                        },
                    }
                    require(
                        candidate_key not in grid_rows
                        or grid_rows[candidate_key] == observation,
                        "Candidate observations disagree between rank policies",
                    )
                    grid_rows[candidate_key] = observation
        require(
            set(target) == {(c, p, b) for c, p in keys for b in policies},
            "Policy coverage incomplete",
        )
    for kind, choices in (("storage", storage), ("rank", ranks)):
        for (core, precision, budget), choice in choices.items():
            selected = choice["status"] == "selected"
            candidate = (
                grid_rows[(core, precision, choice["delta"])] if selected else None
            )
            row = {
                "policy_kind": kind,
                "core": core,
                "precision": precision,
                "logical_cap_bytes": budget if kind == "storage" else None,
                "rank_threshold_fraction": budget if kind == "rank" else None,
                "status": choice["status"],
                "selected_delta": choice["delta"],
                "training_max_logical_bytes": choice["max_training_logical_bytes"],
                "training_max_queried_rank_error_fraction": candidate[
                    "max_queried_training_rank_error"
                ]
                if candidate
                else None,
                "training_cap_slack_bytes": choice.get("training_cap_slack_bytes"),
                "training_cap_violation": False
                if selected and kind == "storage"
                else None,
                "training_rank_policy_violation": False
                if selected and kind == "rank"
                else None,
                "held_out_status": "pending" if selected else "unavailable_within_grid",
                "held_out_cap_violation_states": None,
                "held_out_rank_policy_violation_states": None,
                "held_out_max_logical_bytes": None,
                "held_out_max_queried_rank_error_fraction": None,
                "held_out_max_p50_rank_error_fraction": None,
                "held_out_value_error_by_distribution": None,
                "held_out_modern_cdf_error_by_distribution": None,
                "timing_status": (
                    "unavailable_within_grid"
                    if not selected
                    else "not_in_frozen_timing_subset"
                    if kind == "storage" and budget == 2048
                    else "pending"
                ),
            }
            if not selected:
                best = min(
                    (grid_rows[(core, precision, d)] for d in grid),
                    key=lambda c: c["max_queried_training_rank_error"],
                )
                row["best_grid_delta_diagnostic_only"] = best["delta"]
                row["best_grid_rank_error_fraction_diagnostic_only"] = best[
                    "max_queried_training_rank_error"
                ]
                row["best_grid_threshold_excess_fraction_diagnostic_only"] = (
                    best["max_queried_training_rank_error"] - budget
                )
            rows.append(row)
    return {
        "schema": "training-selection-summary-v1",
        "status": "training_only_held_out_and_timing_pending",
        "selection_lock_sha256": digest,
        "protocol_sha256": PROTOCOL,
        "modern_commit": MODERN,
        "original_commit": ORIGINAL,
        "training_metadata_sha256": data["training_metadata_sha256"],
        "training_raw_files_in_lock": len(data["training_raw_sha256"]),
        "training_states_from_grid_candidate_counts": sum(
            c["states"] for c in grid_rows.values()
        ),
        "scope": "Only selection-lock.json read; raw files and phase metadata were not replayed or independently verified here.",
        "design": design,
        "choices": rows,
        "grid_candidates": list(grid_rows.values()),
    }


def pp(value):
    return "-" if value is None else f"{100 * value:.6f}"


def markdown(summary):
    lines = [
        "# Training selection; held-out results pending",
        "",
        "This summary reads the frozen selection lock only. It is not a held-out accuracy result or a timing result. All 24 choices remain visible; unavailable policies receive no substitute delta.",
        "",
        f"Selection lock SHA256: `{summary['selection_lock_sha256']}`. Modern source: `{MODERN}`. Independent original source: `{ORIGINAL}`.",
        "",
        "All 12 logical-storage policies selected a delta. Eight of the 12 illustrative rank policies were unavailable within the fixed grid: neither core, in either precision, met the 0.5 or 1.0 percentage-point maximum queried training-rank threshold. All four 2.0-point policies selected delta 1200. These thresholds are experimental policies, not product tolerances.",
        "",
        "The largest fitting delta was selected for each storage policy, and the smallest passing delta for each rank policy. Selections use all training datasets and each core's supported routes; the modern route set is broader. Logical bytes count the object and live centroid payload, not retained capacity, allocator overhead, process RSS, or wire size. Lower precision refers to means; modern weights remain f64.",
        "",
        "## All 24 frozen choices",
        "",
        "Rank errors below are maximum queried training errors in percentage points. Cap and rank policies are separate: passing a storage cap does not mean passing an accuracy threshold. Dashes denote unavailable or inapplicable values; held-out fields have not been measured.",
        "",
        "| Policy | Core | Mean precision | Status | Delta | Training max bytes | Cap slack bytes | Training max rank error (pp) | Held-out |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["choices"]:
        policy = (
            f"{row['logical_cap_bytes']} B"
            if row["policy_kind"] == "storage"
            else f"{pp(row['rank_threshold_fraction'])} pp"
        )
        values = [
            policy,
            row["core"],
            row["precision"],
            row["status"],
            row["selected_delta"],
            row["training_max_logical_bytes"],
            row["training_cap_slack_bytes"],
            pp(row["training_max_queried_rank_error_fraction"]),
            row["held_out_status"],
        ]
        lines.append(
            "| " + " | ".join("-" if x is None else str(x) for x in values) + " |"
        )
    lines += [
        "",
        "## What the training result establishes",
        "",
        "The storage policies choose smaller modern deltas than original deltas at every common cap and precision. That makes the actual accuracy tradeoff important; common caps do not establish equal accuracy or equal memory use. The coarse delta grid also leaves slack: original f32 uses at most 2576 of 4096 bytes, while delta 600 needs 4688 bytes. No storage choice is at the grid's delta-1200 upper edge.",
        "",
        "At the separate 2.0-point rank policy, all four choices use delta 1200. The maximum training logical sizes are 13248/8848 bytes for original f64/f32 and 26504/17776 bytes for modern f64/f32. Their observed maximum training errors differ, so this does not establish equal error, an optimal parameter choice, or a universal error bound.",
        "",
        "## Pending result tables",
        "",
        "The JSON/CSV scaffold reserves distinct held-out cap-miss and rank-miss counts, maximum logical bytes, queried rank and p50 rank errors, per-distribution value errors, and modern-only CDF errors. Pending values must remain null until supported by completed phase evidence. Unavailable policies remain unavailable after held-out evaluation.",
        "",
        "The completed report should show route and distribution breakdowns, preserve all cap/rank violations and inherited original quirks, and identify logical bytes separately from retained capacity/RSS and wire size. The selected-delta timing subset is governed by its separately frozen protocol; its n=1M inputs extrapolate beyond this training matrix's n<=100K. Timing ratios require completed controls and cannot be inferred from this table.",
        "",
        "## Compact review bundle",
        "",
        "Keep the protocol and helpers, full compact selection lock, phase metadata/ledgers, raw-file hash manifests, aggregated JSON/CSV, and a few explicitly identified worst-case records in the review bundle. Retain bulk compressed raw output in the scratch study directory rather than Git. The raw manifests alone support identity, not reconstruction; provide rerun commands and an external archive location/hash if the bulk data is shared later.",
        "",
        "This summary verified the lock SHA, exact source/protocol identities, all 24 policy keys, every candidate grid, and selection/slack arithmetic from the lock. It did not replay raw data or independently verify phase metadata. The full timing runner still requires its stronger completed-phase and raw-evidence guards.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.lock.read_bytes()
    summary = summarize(json.loads(raw), hashlib.sha256(raw).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection-lock.json").write_bytes(raw)
    (args.output / "selection-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "TRAINING_SELECTION.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    csv_buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in summary["choices"] for key in row))
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary["choices"])
    (args.output / "selection-summary.csv").write_text(
        csv_buffer.getvalue(), encoding="utf-8", newline=""
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": len(summary["choices"]),
                "grid_candidates": len(summary["grid_candidates"]),
                "states_from_lock": summary[
                    "training_states_from_grid_candidate_counts"
                ],
                "selection_lock_sha256": summary["selection_lock_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
