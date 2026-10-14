#!/usr/bin/env python3
"""Read completed controlled results; do not rerun any measured operation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.resolve()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["mode"] == "timing"
    assert metadata["completed_processes"] == 32
    assert not metadata["numerical_state_mismatches"]
    assert not any(metadata["source_changes"].values())
    assert sha(folder / "harness.rs") == metadata["focused_harness_sha256"]
    for label, snapshot in metadata["snapshots"].items():
        source = folder / "sources" / label
        assert sha(source / "Cargo.toml") == snapshot["cargo_manifest_sha256"]
        assert sha(source / "Cargo.lock") == snapshot["cargo_lock_sha256"]
        for name, expected in snapshot["source_sha256"].items():
            assert sha(source / name) == expected, (label, name)
    original_provenance = json.loads((folder / "sources/baseline/compat/performance/original-source-provenance.json").read_text())
    for name, expected in original_provenance["normalized_lf_sha256"].items():
        assert sha(folder / "upstream-core/src" / name) == expected, name
    for label in metadata["versions"]:
        assert sha(folder / "harness-crates" / label / "Cargo.lock") == metadata["resolved_lock_sha256"]
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert Counter(r["version"] for r in rows) == {"baseline": 16, "candidate": 16}
    groups = defaultdict(list)
    for row, scheduled in zip(rows, metadata["planned_runs"]):
        case = (row["kind"], row["n"], row["precision"])
        assert case == (scheduled["kind"], scheduled["n"], scheduled["precision"])
        assert (row["round"], row["schedule_position"], row["version"]) == (
            scheduled["round"], scheduled["position"], scheduled["version"]
        )
        assert row["original_first"] == (scheduled["order"] == "original-first")
        groups[case].append(row)
    for label, binary in metadata["binaries"].items():
        assert sha(Path(binary["path"])) == binary["sha256"], label
    for case, values in groups.items():
        assert len(values) == 8
        assert all(r["state"] == values[0]["state"] for r in values)
    operations = ["original_build", "modern_build", "original_p50", "modern_p50"]
    comparison = []
    for case, values in groups.items():
        for operation in operations:
            medians = {version: [r[operation]["median_ns"] for r in values if r["version"] == version]
                       for version in ["baseline", "candidate"]}
            by_round = []
            for round_index in [0, 1]:
                block = {version: statistics.median(r[operation]["median_ns"] for r in values
                         if r["version"] == version and r["round"] == round_index)
                         for version in medians}
                by_round.append(block["candidate"] / block["baseline"])
            comparison.append({
                "case": list(case), "operation": operation,
                "run_medians_ns": medians,
                "baseline_median_ns": statistics.median(medians["baseline"]),
                "candidate_median_ns": statistics.median(medians["candidate"]),
                "candidate_over_baseline": statistics.median(medians["candidate"]) / statistics.median(medians["baseline"]),
                "block_candidate_over_baseline": by_round,
            })
    elapsed = (datetime.fromisoformat(metadata["finished_utc"]) - datetime.fromisoformat(metadata["started_utc"])).total_seconds()
    report = {
        "status": "complete; controls independently rechecked",
        "source_versions": metadata["versions"],
        "processes": len(rows), "identical_state_cases": len(groups),
        "elapsed_including_build_seconds": elapsed,
        "timed_process_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
        "source_changes": metadata["source_changes"], "state_mismatches": [],
        "comparisons": comparison,
        "scope": "Descriptive process medians and two counterbalanced blocks on one machine/seed; no significance or causality claim",
        "artifact_sha256": {name: sha(folder / name) for name in ["metadata.json", "results.jsonl", "harness.rs"]},
    }
    (folder / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Controlled interleaved comparison: 22867bd versus c5dec35",
        "",
        "All 32 processes completed. Exact original/modern query summaries, centroid",
        "state, modern count/sum bits and complete modern wire bytes matched across",
        "both versions and all repeats for each of the four cases. Source files stayed",
        "unchanged, manifests/locks matched and retained executable hashes verify.",
        "",
        f"The run occupied {elapsed / 60:.1f} minutes including its release builds;",
        f"measured processes account for {report['timed_process_elapsed_seconds'] / 60:.1f} minutes.",
        "",
        "## Descriptive timing comparison",
        "",
        "Ratios are candidate/baseline; values above 1 mean the candidate took longer.",
        "The main ratio compares the median of four process medians per version. Each",
        "block ratio compares two process medians per version. These are descriptive",
        "summaries, not independent-sample confidence intervals.",
        "",
        "| Input | Operation | Baseline | Candidate | Ratio | ABBA block | BAAB block |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in comparison:
        kind, n, precision = item["case"]
        scale, unit = (1e9, "s") if n == 10000000 and item["operation"].endswith("build") else (1e3, "us")
        lines.append(f"| {kind} {n:,} {precision.upper()} | {item['operation']} | "
                     f"{item['baseline_median_ns']/scale:.3f} {unit} | {item['candidate_median_ns']/scale:.3f} {unit} | "
                     f"{item['candidate_over_baseline']:.3f} | {item['block_candidate_over_baseline'][0]:.3f} | "
                     f"{item['block_candidate_over_baseline'][1]:.3f} |")
    lines.extend([
        "", "## Method and limits", "",
        "The immutable source commits are", f"`{metadata['versions']['baseline']}` and",
        f"`{metadata['versions']['candidate']}`. The independent original is",
        f"`{metadata['original_pin']}`. The newer transactional-scaling/packaging",
        "candidate was not a measured arm and must not be silently substituted in this record.", "",
        "Release builds used Rust 1.90.0 and the same resolved dependencies/allocator.",
        f"Each process ran on Linux logical CPU {metadata['affinity_cpu']} with RAYON_NUM_THREADS=1.",
        "Round one used ABBA and original-first operation ordering; round two used",
        "BAAB, reversed cases and modern-first ordering. Each operation retained three",
        "warmups, 15 ms calibration and seven batches targeting 25 ms. The inherited",
        "timer sorts its seven sample values, so their chronological order is unavailable.", "",
        "The unchanged original core is an environmental control, not a correction",
        "factor that can erase drift. The host was not thermally or physically isolated.",
        "Only two blocks, one seed, four cases and one machine are covered. Consistency",
        "across blocks warrants investigation; it does not identify the responsible",
        "instruction or establish a general performance ranking. Full raw process",
        "records, source exports, binaries, logs and metadata are retained beside this report.",
        "",
    ])
    (folder / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in report.items() if key not in ["comparisons", "artifact_sha256"]}, indent=2))
    for item in comparison:
        print(f"{item['case']} {item['operation']}: ratio={item['candidate_over_baseline']:.4f} blocks={item['block_candidate_over_baseline']}")


if __name__ == "__main__":
    main()
