"""Package verified compact reducer outputs; never open raw compressed cases."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

from summarize_selection import MODERN, ORIGINAL, require


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def maximum(rows, key):
    values = [r[key] for r in rows]
    return max(values) if values and all(v is not None for v in values) else None


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def pp(value):
    return "-" if value is None else f"{100 * value:.4f}"


def number(value):
    return "-" if value is None else f"{value:.6g}"


def route_class(route):
    return (
        "direct"
        if route == "direct"
        else "kway"
        if route.endswith("/kway")
        else "sequential"
    )


def cell_name(row):
    return (
        f"{row['logical_cap_bytes']} B"
        if row["policy_kind"] == "storage"
        else f"{100 * row['rank_threshold_fraction']:g} pp"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reduced, output = args.reduced.resolve(), args.output.resolve()
    require(not output.exists(), "Fresh bundle directory required")
    status = read(reduced / "status.json")
    require(
        status["status"] == "complete" and not status["failures"],
        "Reducer did not complete cleanly",
    )
    require(
        all(
            digest(reduced / name) == sha
            for name, sha in status["output_sha256"].items()
        ),
        "Reduced output hash differs",
    )
    summary = read(reduced / "summary.json")
    with (reduced / "state-metrics.jsonl").open() as stream:
        states = [json.loads(line) for line in stream]
    with (reduced / "route-pairs.jsonl").open() as stream:
        pairs = [json.loads(line) for line in stream]
    require(
        len(states) == 8064 and len(pairs) == 5184, "Compact record coverage differs"
    )
    output.mkdir(parents=True)
    results = output / "results"
    results.mkdir()
    policy_rows = []
    for choice in summary["choices"]:
        metrics = choice["full_metrics"]
        row = {
            "policy": cell_name(choice),
            "policy_kind": choice["policy_kind"],
            "cap_bytes": choice["logical_cap_bytes"],
            "rank_threshold_fraction": choice["rank_threshold_fraction"],
            "core": choice["core"],
            "precision": choice["precision"],
            "status": choice["status"],
            "delta": choice["selected_delta"],
            "heldout_states": choice["held_out_states"],
            "policy_violation_states": choice["held_out_policy_violation_states"],
            "max_logical_bytes": choice["held_out_max_logical_bytes"],
            "max_rank_error_fraction": metrics["max_rank_error"]["max"]
            if metrics
            else None,
            "max_p50_rank_error_fraction": metrics["p50_rank_error"]["max"]
            if metrics
            else None,
            "max_cdf_error_fraction": metrics.get("max_cdf_error", {}).get("max")
            if metrics
            else None,
        }
        policy_rows.append(row)
    csv_file(results / "policy-outcomes.csv", policy_rows)
    groups = defaultdict(list)
    for row in states:
        groups[
            (
                row["policy"],
                row["threshold"],
                row["precision"],
                row["core"],
                row["kind"],
                route_class(row["route"]),
            )
        ].append(row)
    distribution_rows, tails = [], []
    for key, rows in sorted(groups.items()):
        identity = dict(
            zip(
                (
                    "policy_kind",
                    "threshold",
                    "precision",
                    "core",
                    "distribution",
                    "route_class",
                ),
                key,
                strict=True,
            )
        )
        distribution_rows.append(
            {
                **identity,
                "states": len(rows),
                "sizes": ";".join(map(str, sorted({r["n"] for r in rows}))),
                "fresh_seeds": ";".join(map(str, sorted({r["seed"] for r in rows}))),
                "max_rank_error_fraction": maximum(rows, "max_rank_error"),
                "max_p50_rank_error_fraction": maximum(rows, "p50_rank_error"),
                "max_p50_abs_value_error": maximum(rows, "p50_abs_type7_error"),
                "max_abs_value_error": maximum(rows, "max_abs_type7_error"),
                "max_cdf_error_fraction": maximum(rows, "max_cdf_error")
                if key[3] == "modern"
                else None,
                "max_logical_bytes": maximum(rows, "logical_bytes"),
                "policy_violations": sum(r["policy_violation"] for r in rows),
            }
        )
        for q in (0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999):
            points = [
                next(t for t in row["tails"] if t["requested_q"] == q) for row in rows
            ]
            tails.append(
                {
                    **identity,
                    "requested_q": q,
                    "effective_q": points[0]["effective_q"],
                    "states": len(rows),
                    "max_rank_error_fraction": maximum(points, "rank_error"),
                    "max_abs_value_error": maximum(points, "abs_value_error"),
                }
            )
    csv_file(results / "distribution-route-accuracy.csv", distribution_rows)
    csv_file(results / "tail-accuracy.csv", tails)
    route_rows = []
    for group in summary["groups"]:
        identity = group["identity"]
        route_rows.append(
            {
                **identity,
                "fresh_seeds": ";".join(map(str, group["seeds"])),
                "policy_violation_states": group["policy_violation_states"],
                **{
                    f"max_{name}": value["max"]
                    for name, value in group["metrics"].items()
                },
                **{
                    f"missing_{name}": value["nonfinite_or_missing"]
                    for name, value in group["metrics"].items()
                },
            }
        )
    csv_file(results / "per-route-size-seed-group.csv", route_rows)
    paired_groups = defaultdict(list)
    for pair in pairs:
        paired_groups[
            tuple(
                pair[k]
                for k in (
                    "policy",
                    "threshold",
                    "precision",
                    "kind",
                    "comparison",
                    "modern_route",
                )
            )
        ].append(pair)
    pair_rows = []
    for key, rows in sorted(paired_groups.items()):
        pair_rows.append(
            {
                **dict(
                    zip(
                        (
                            "policy",
                            "threshold",
                            "precision",
                            "distribution",
                            "comparison",
                            "modern_route",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "states": len(rows),
                "p50_rank_lower": sum(
                    r["p50_rank_error_change"] is not None
                    and r["p50_rank_error_change"] < 0
                    for r in rows
                ),
                "p50_rank_equal": sum(r["p50_rank_error_change"] == 0 for r in rows),
                "p50_rank_higher": sum(
                    r["p50_rank_error_change"] is not None
                    and r["p50_rank_error_change"] > 0
                    for r in rows
                ),
                "max_rank_lower": sum(
                    r["max_rank_error_change"] is not None
                    and r["max_rank_error_change"] < 0
                    for r in rows
                ),
                "max_rank_equal": sum(r["max_rank_error_change"] == 0 for r in rows),
                "max_rank_higher": sum(
                    r["max_rank_error_change"] is not None
                    and r["max_rank_error_change"] > 0
                    for r in rows
                ),
                "missing_comparisons": sum(
                    r["max_rank_error_change"] is None
                    or r["p50_rank_error_change"] is None
                    for r in rows
                ),
                "min_actual_logical_bytes_ratio": min(
                    r["logical_bytes_ratio"] for r in rows
                ),
                "max_actual_logical_bytes_ratio": max(
                    r["logical_bytes_ratio"] for r in rows
                ),
            }
        )
    csv_file(results / "paired-route-comparisons.csv", pair_rows)
    notable = {}

    def retain(rows, metric, reason):
        if not rows:
            return
        row = max(
            rows, key=lambda r: r[metric] if r[metric] is not None else float("inf")
        )
        key = (row["source_case_file"], row["source_record_1based"])
        notable.setdefault(key, {"reasons": [], "state": row})["reasons"].append(reason)

    for core in ("original", "modern"):
        for precision in ("f64", "f32"):
            subset = [
                r for r in states if r["core"] == core and r["precision"] == precision
            ]
            for policy in ("storage", "rank"):
                selected = [r for r in subset if r["policy"] == policy]
                for metric in ("max_rank_error", "p50_rank_error") + (
                    ("max_cdf_error",) if core == "modern" else ()
                ):
                    retain(
                        selected, metric, f"worst {policy}/{core}/{precision}/{metric}"
                    )
            for distribution in sorted({r["kind"] for r in subset}):
                selected = [
                    r
                    for r in subset
                    if r["kind"] == distribution and r["policy"] == "rank"
                ]
                retain(
                    selected,
                    "p50_abs_type7_error",
                    f"2pp policy worst p50 value: {distribution}/{core}/{precision}",
                )
    (results / "notable-states.json").write_text(
        json.dumps(list(notable.values()), indent=2, allow_nan=False) + "\n"
    )
    shutil.copy2(reduced / "status.json", results / "reducer-status.json")
    shutil.copy2(reduced / "raw-file-sha256.json", results / "raw-file-sha256.json")
    source_dir = Path(__file__).parent
    for name in (
        "reduce_heldout.py",
        "test_reducer.py",
        "summarize_heldout_metadata.py",
        "summarize_selection.py",
        "package_budget_results.py",
        "HELD_OUT_REPORTING.md",
        "STREAMING_REDUCER.md",
        "REDUCER_READY.json",
        "reducer-synthetic-checks.log",
    ):
        shutil.copy2(source_dir / name, output / name)
    protocol_dir = output / "protocol"
    protocol_dir.mkdir()
    for path in args.protocol.iterdir():
        if path.is_file() and path.suffix in (".py", ".rs", ".json", ".md"):
            shutil.copy2(path, protocol_dir / path.name)
    shutil.copy2(
        source_dir / "selected-training/selection-lock.json",
        protocol_dir / "selection-lock.json",
    )
    storage_worst = max(
        (r for r in states if r["core"] == "modern" and r["policy"] == "storage"),
        key=lambda r: r["max_rank_error"],
    )
    cdf_worst = max(
        (r for r in states if r["core"] == "modern" and r["policy"] == "rank"),
        key=lambda r: r["max_cdf_error"],
    )
    median_worst = max(
        (r for r in states if r["core"] == "modern" and r["policy"] == "rank"),
        key=lambda r: r["p50_rank_error"],
    )
    lines = [
        "# Accuracy at frozen logical-storage and rank-error policies",
        "",
        f"Modern source `{MODERN}`; independent original `{ORIGINAL}`. These measurements compare the opt-in modern core configured with K2Norm and singleton policy Off against the default original algorithm. Compatibility API parity is assessed separately.",
        "",
        "Training selected deltas globally before the three fresh held-out seeds were evaluated. All 576 held-out dataset-policy invocations and 8064 states completed, with no modern invariant/wire failures. The offline raw-data reducer verified hashes, paired query/reference identities, error arithmetic and policy outcomes in one pass; it did not independently regenerate every large-n reference sample. It evaluated 8088192 quantile estimates; modern CDF probe count is recorded below. No operation timing is included.",
        "",
        f"All 6048 storage-policy states met their selected cap, and all 2016 states under the selected 2.0 percentage-point rank policy met that threshold. The 0.5 and 1.0-point policies remain unavailable within the frozen delta 25–1200 grid for both cores and both precisions. Passing a storage cap does not imply small rank error: the worst modern storage-policy queried error was **{pp(storage_worst['max_rank_error'])} percentage points** ({storage_worst['threshold']} B, {storage_worst['precision']}, {storage_worst['kind']}, n={storage_worst['n']}, {storage_worst['route']}).",
        "",
        "## All frozen policies",
        "",
        "All errors in this table are worst observed percentage points across the queried grid and the core's supported routes. Modern coverage includes additional k-way routes. Rank-policy thresholds and storage caps are separate; logical bytes are object plus live centroid payload, not capacity, peak memory or RSS.",
        "",
        "| Policy | Core | Means | Delta | States | Misses | Max bytes | Max rank (pp) | Max p50 rank (pp) | Modern CDF (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in policy_rows:
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    (
                        r["policy"],
                        r["core"],
                        r["precision"],
                        r["delta"] if r["delta"] is not None else "unavailable",
                        r["heldout_states"],
                        r["policy_violation_states"]
                        if r["policy_violation_states"] is not None
                        else "-",
                        r["max_logical_bytes"]
                        if r["max_logical_bytes"] is not None
                        else "-",
                        pp(r["max_rank_error_fraction"]),
                        pp(r["max_p50_rank_error_fraction"]),
                        pp(r["max_cdf_error_fraction"]),
                    ),
                )
            )
            + " |"
        )
    lines += [
        "",
        "## What the matched policies show",
        "",
        "The modern storage choices use smaller deltas at every common cap. At the 2.0-point rank policy all four choices use delta 1200, but modern f64/f32 reach 26504/17760 logical bytes versus original 13248/8848. Both meet the same illustrative threshold on these probes; they do not have equal actual error or equal memory use. None of this establishes an optimal parameter choice or a universal bound.",
        "",
        f"At that rank policy, worst modern p50 rank error is {pp(median_worst['p50_rank_error'])} pp and worst modern CDF error is {pp(cdf_worst['max_cdf_error'])} pp. CDF has its own midpoint-ECDF metric and no original API equivalent, so the 2.0-point quantile policy does not certify a CDF limit.",
        "",
        "A small rank error can correspond to a large value error across the empty gap between separated modes. Conversely, a quantile at an exact tied value can have zero tie-aware rank error while differing from an interpolating Type 7 reference. Both metrics are retained for this reason. These modern-core results do not justify a universal tiny-epsilon change relative to the original; existing compatibility behavior is a separate contract.",
        "",
        "## Distribution detail at the separate 2.0-point rank policy",
        "",
        "The following rows combine the available routes/sizes within each distribution and precision, over the same three fresh seeds. Value error is in that distribution's input units and is not pooled across distributions. Detailed route/sizing and six fixed tail queries are in the CSV files.",
        "",
        "| Distribution | Core | Means | Max rank (pp) | Max p50 rank (pp) | Max p50 value error | Max value error, all q | Modern CDF (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for distribution in sorted({r["kind"] for r in states}):
        for core in ("original", "modern"):
            for precision in ("f64", "f32"):
                rows = [
                    r
                    for r in states
                    if r["policy"] == "rank"
                    and r["kind"] == distribution
                    and r["core"] == core
                    and r["precision"] == precision
                ]
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            distribution,
                            core,
                            precision,
                            pp(maximum(rows, "max_rank_error")),
                            pp(maximum(rows, "p50_rank_error")),
                            number(maximum(rows, "p50_abs_type7_error")),
                            number(maximum(rows, "max_abs_type7_error")),
                            pp(maximum(rows, "max_cdf_error"))
                            if core == "modern"
                            else "-",
                        )
                    )
                    + " |"
                )
    lines += [
        "",
        "## Routes, references and limits",
        "",
        "There are 2880 same-route original/modern comparisons and 2304 separately labelled modern-k-way/original-sequential comparisons. The CSV comparison counts are descriptive correlated state observations, not independent samples or significance tests. Direct, balanced-IID/range and skewed 40-way merge layouts remain available separately; reverse shard order is present only at n=100000.",
        "",
        "Rank error measures distance from effective q to the empirical tie interval [F(x-), F(x)]. Value error uses Type 7 quantiles of post-cast samples. Modern CDF error uses the midpoint of the empirical jump. Both cores receive the same mean precision and representable q. Requested/effective tail queries and p50 estimates are retained in compact full results and selected notable states. No blanket epsilon or world-best claim follows from these inputs.",
        "",
        "The independent original records 1152 out-of-bounds/nonfinite-quantile observation flags and 1152 decreasing-query flags. These are retained state occurrences, including repeated baseline evaluations across policies/routes, not 1152 independent bad datasets; the combined flag does not assert nonfinite values specifically. Original accuracy remains visible despite inherited semantics.",
        "",
        "Fresh seeds are 15485863, 32452843 and 49979687; training used 1009, 10007 and 65537. Sizes are 32, 1000 and 100000, with eight distributions and unit weights. There is no claim about all production distributions, fractional weights, other scale/policy settings, equal RSS, or speed. The separate timing experiment uses n=1M and reports its own cap/error violations.",
        "",
        "## Evidence and reproduction",
        "",
        "[Policy outcomes](results/policy-outcomes.csv) includes every choice; [distribution/route accuracy](results/distribution-route-accuracy.csv) and [tail accuracy](results/tail-accuracy.csv) preserve per-distribution metrics; [route/size/seed groups](results/per-route-size-seed-group.csv) resolves each route, size and three-seed group; [paired route comparisons](results/paired-route-comparisons.csv) separates algorithms. [Notable states](results/notable-states.json) identifies selected extrema with raw filenames/hashes/record numbers. Compiled source identities and raw hashes remain in [reducer status](results/reducer-status.json) and the [raw manifest](results/raw-file-sha256.json).",
        "",
        "[The bundle guide](README.md) labels historical preparation documents and describes the current evidence. `protocol/` contains the frozen runner/selector/reference sources and selection lock. Reproduction requires the recorded source commits and build, smoke, training, selection and held-out phases in order, then reporting. Bulk compressed raw data, complete compact state records and binaries remain outside Git at the paths in [full result locations](FULL_RESULTS_LOCATION.json); hashes verify identity, not availability. No bulk raw samples are included in this bundle.",
        "",
        f"Reduction status: complete; {status['decompressed_bytes']} expanded bytes; {summary['quantile_probes']} quantile estimates and {summary['modern_cdf_probes']} modern CDF probes. Reduction wall time {status['elapsed_seconds']:.3f} seconds is offline raw-data processing, not a library performance measurement.",
        "",
    ]
    require(summary["quantile_probes"] == 8088192, "Quantile count text differs")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "README.md").write_text(
        "# Matched-budget accuracy evidence\n\n"
        "Read [RESULTS.md](RESULTS.md) for the completed, source-labelled results and [PACKAGE-MANIFEST.json](PACKAGE-MANIFEST.json) for bundle hashes. The measured configuration is opt-in modern K2Norm with singleton policy Off versus the default original algorithm. This bundle contains no timing result.\n\n"
        "[HELD_OUT_REPORTING.md](HELD_OUT_REPORTING.md), [STREAMING_REDUCER.md](STREAMING_REDUCER.md), [REDUCER_READY.json](REDUCER_READY.json), and protocol [PLAN.md](protocol/PLAN.md)/[REVIEW.md](protocol/REVIEW.md) are historical preparation records retained byte-for-byte. Their pending/not-executed wording describes the time they were written; it is superseded by the completed [reducer status](results/reducer-status.json). Frozen protocol/source hashes must not be changed to update historical prose.\n\n"
        "The [metadata-only policy report](results/metadata-policy-summary.json) preceded the offline full raw-data reduction. Its intentionally missing numerical fields are supplied by the CSVs linked from RESULTS.md. [Full result locations](FULL_RESULTS_LOCATION.json) identifies retained bulk evidence outside this compact bundle.\n",
        encoding="utf-8",
    )
    shutil.copy2(source_dir / "held-out-metadata-ff73cfa/policy-summary.json", results / "metadata-policy-summary.json")
    location = {
        "modern_commit": MODERN,
        "original_commit": ORIGINAL,
        "full_reduced_directory": str(reduced),
        "full_study_directory": "/var/tmp/codex-tdigest-pr3-20261014/overnight/storage-budget-study-ff73cfa",
        "complete_reduced_output_sha256": status["output_sha256"],
        "bulk_raw_included": False,
        "compact_full_records_included": False,
        "reason": "Keep review small; full data and compact states retained in scratch with hashes and reproducible protocol.",
    }
    (output / "FULL_RESULTS_LOCATION.json").write_text(
        json.dumps(location, indent=2) + "\n"
    )
    manifest = {
        "schema": "budget-accuracy-review-bundle-v1",
        "status": "complete",
        "source_reducer_status_sha256": digest(reduced / "status.json"),
        "files_sha256": {
            str(p.relative_to(output)).replace("\\", "/"): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "PACKAGE-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "bundle": str(output),
                "bundle_files": len(manifest["files_sha256"]) + 1,
                "bundle_bytes": sum(
                    p.stat().st_size for p in output.rglob("*") if p.is_file()
                ),
                "output_sha256": status["output_sha256"],
                "worst_modern_storage_rank": {
                    k: storage_worst[k]
                    for k in (
                        "threshold",
                        "precision",
                        "kind",
                        "n",
                        "seed",
                        "route",
                        "max_rank_error",
                    )
                },
                "worst_modern_rank_policy_p50": {
                    k: median_worst[k]
                    for k in (
                        "precision",
                        "kind",
                        "n",
                        "seed",
                        "route",
                        "p50_rank_error",
                        "p50_abs_type7_error",
                    )
                },
                "worst_modern_rank_policy_cdf": {
                    k: cdf_worst[k]
                    for k in (
                        "precision",
                        "kind",
                        "n",
                        "seed",
                        "route",
                        "max_cdf_error",
                    )
                },
            }
        )
    )


if __name__ == "__main__":
    main()
