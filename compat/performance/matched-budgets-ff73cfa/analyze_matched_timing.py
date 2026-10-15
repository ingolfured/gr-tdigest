"""Summarize a completed frozen timing run; never execute native code."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import run_timing as runner


SEQ = "balanced_iid/forward/sequential"
KWAY = "balanced_iid/forward/kway"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def label(policies):
    return "; ".join(
        f"{p['threshold']} B"
        if p["policy"] == "storage"
        else f"{100 * p['threshold']:g} pp"
        for p in policies
    )


def max_finite(values):
    return max(values) if values and all(v is not None for v in values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--independent-review", type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    require = runner.require
    require(not output.exists(), "Fresh output required")
    build, meta = read(run / "build.json"), read(run / "full/metadata.json")
    require(
        digest(run / "build.json") == (run / "build.sha256").read_text().strip(),
        "Build metadata seal differs",
    )
    require(
        digest(run / "full/metadata.json")
        == (run / "full/metadata.sha256").read_text().strip(),
        "Full metadata seal differs",
    )
    require(
        build["status"] == meta["status"] == "complete" and not meta["failures"],
        "Timing not complete",
    )
    require(
        build["modern_commit"] == runner.MODERN
        and build["original_commit"] == runner.ORIGINAL,
        "Sources differ",
    )
    require(
        len(build["pairs"]) == 18
        and len(meta["planned_runs"]) == len(meta["completed_runs"]) == 72,
        "Coverage differs",
    )
    require(
        meta["planned_runs"] == runner.schedule(build["pairs"], False),
        "Counterbalanced schedule differs",
    )
    for name, expected in build["implementation_sha256"].items():
        require(
            digest(Path(__file__).parent / name) == expected,
            "Frozen implementation differs: " + name,
        )
    require(
        digest(run / "timing") == build["binary_sha256"] == meta["binary_sha256"],
        "Binary differs",
    )
    require(
        digest(run / "harness/Cargo.lock") == build["resolved_lock_sha256"],
        "Lock differs",
    )
    require(
        digest(run / "selection-lock.json")
        == build["selection_sha256"]
        == meta["selection_sha256"],
        "Selection differs",
    )
    for name, expected in meta["raw_sha256"].items():
        require(digest(run / "full" / name) == expected, "Raw result differs: " + name)
    output.mkdir(parents=True)
    results = output / "results"
    results.mkdir()
    times, state_metrics, controls, input_ids = {}, {}, {}, {}
    batches, tails, policy_evaluations, compact_controls = [], [], [], []
    inherited = Counter()
    for index, (case, done) in enumerate(
        zip(meta["planned_runs"], meta["completed_runs"], strict=True)
    ):
        require(
            case == done["case"] and done["execution"]["exit_code"] == 0,
            "Run identity/status differs",
        )
        raw = read(run / "full" / done["stdout"])
        require(
            runner.validate_result(raw, case) == done["policy_evaluations"],
            "Policy evaluations differ",
        )
        pair, core, block = case["pair_index"], case["core"], case["block"]
        query_controls = sorted(
            [
                {
                    k: t[k]
                    for k in (
                        "operation",
                        "route",
                        "checksum_bits",
                        "expected_checksum_bits",
                        "query_input_bits",
                        "query_value_bits",
                    )
                }
                for t in raw["timings"]
                if t["operation"] in ("p50", "cdf")
            ],
            key=lambda t: (t["operation"], t["route"]),
        )
        control = {
            k: raw[k] for k in ("input_fnv1a64", "leaf_states", "states", "shard_sizes")
        }
        control["query_controls"] = query_controls
        control_sha = identity_hash(control)
        require(
            (pair, core) not in controls or controls[(pair, core)] == control_sha,
            "Same-core controls changed across blocks",
        )
        controls[(pair, core)] = control_sha
        require(
            pair not in input_ids or input_ids[pair] == raw["input_fnv1a64"],
            "Original/modern inputs differ",
        )
        input_ids[pair] = raw["input_fnv1a64"]
        compact_controls.append(
            {
                "pair": pair,
                "core": core,
                "block": block,
                "raw_file": done["stdout"],
                "raw_sha256": meta["raw_sha256"][done["stdout"]],
                "input_fnv1a64": raw["input_fnv1a64"],
                "state_leaf_query_control_sha256": control_sha,
                "query_controls": query_controls,
                "leaf_centroid_counts": [r["count"] for r in raw["leaf_states"]],
                "leaf_logical_bytes": [r["logical_bytes"] for r in raw["leaf_states"]],
            }
        )
        base = {
            "pair": pair,
            "policy": label(case["policies"]),
            "kind": case["kind"],
            "precision": case["precision"],
            "original_delta": case["original_delta"],
            "modern_delta": case["modern_delta"],
        }
        for timing in raw["timings"]:
            times[(pair, core, block, timing["operation"], timing["route"])] = (
                statistics.median(timing["ns_per_operation"])
            )
            batches.append({**base, "core": core, "block": block, **timing})
        for state in raw["states"]:
            details = state["state"]
            qs = details["quantile"]["rows"]
            p50 = next(q for q in qs if q[0] == 0.5)
            cdf = details.get("cdf", {}).get("rows", [])
            evaluation = [
                r for r in done["policy_evaluations"] if r["route"] == state["route"]
            ]
            for item in evaluation:
                policy_evaluations.append({**base, "block": block, **item})
            inherited.update(state["inherited_original_observations"])
            item = {
                **base,
                "core": core,
                "route": state["route"],
                "n": case["n"],
                "seed": case["seed"],
                "centroids": details["centroids"],
                "logical_bytes": details["logical_retained_bytes"],
                "actual_vector_capacity_bytes": details["actual_vector_capacity_bytes"],
                "modern_wire_bytes": details.get("wire_bytes"),
                "max_rank_error_fraction": max_finite([q[4] for q in qs]),
                "p50_rank_error_fraction": p50[4],
                "p50_estimate": p50[2],
                "p50_type7_reference": p50[3],
                "p50_abs_value_error": p50[5],
                "max_abs_value_error": max_finite([q[5] for q in qs]),
                "max_cdf_error_fraction": max_finite([q[3] for q in cdf]),
                "policy_violation": any(r["violation"] for r in evaluation),
                "max_policy_excess": max(r["excess"] or 0 for r in evaluation),
                "inherited_original_observations": "; ".join(
                    state["inherited_original_observations"]
                ),
            }
            key = (pair, core, state["route"])
            require(
                key not in state_metrics or state_metrics[key] == item,
                "State metrics changed across blocks",
            )
            state_metrics[key] = item
            if block == 1:
                for requested in (0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999):
                    q = next(q for q in qs if q[0] == requested)
                    tails.append(
                        {
                            **base,
                            "core": core,
                            "route": state["route"],
                            **dict(
                                zip(
                                    (
                                        "requested_q",
                                        "effective_q",
                                        "estimate",
                                        "type7_reference",
                                        "rank_error_fraction",
                                        "abs_value_error",
                                    ),
                                    q,
                                    strict=True,
                                )
                            ),
                        }
                    )
    require(
        len(state_metrics) == 90 and len(batches) == 468 and len(controls) == 36,
        "Final coverage differs",
    )
    paired_rows, kway_rows, cdf_rows = [], [], []
    for pair, selected in enumerate(build["pairs"]):
        base = {
            "pair": pair,
            "policy": label(selected["policies"]),
            **{
                k: selected[k]
                for k in ("kind", "precision", "original_delta", "modern_delta")
            },
        }
        for operation, route in (
            ("build", "direct"),
            ("sequential", SEQ),
            ("p50", "direct"),
            ("p50", SEQ),
        ):
            o1, o2 = [times[(pair, "original", b, operation, route)] for b in (1, 2)]
            m1, m2 = [times[(pair, "modern", b, operation, route)] for b in (1, 2)]
            paired_rows.append(
                {
                    **base,
                    "operation": operation,
                    "route": route,
                    "original_b1_ns": o1,
                    "modern_b1_ns": m1,
                    "modern_over_original_b1": m1 / o1,
                    "original_b2_ns": o2,
                    "modern_b2_ns": m2,
                    "modern_over_original_b2": m2 / o2,
                    "original_b2_over_b1": o2 / o1,
                    "modern_b2_over_b1": m2 / m1,
                    "original_policy_violation": state_metrics[
                        (pair, "original", route)
                    ]["policy_violation"],
                    "modern_policy_violation": state_metrics[(pair, "modern", route)][
                        "policy_violation"
                    ],
                }
            )
        for operation in ("kway", "p50"):
            row = {
                **base,
                "operation": operation,
                "comparison": "modern kway versus modern/original sequential; different routes",
            }
            for block in (1, 2):
                k = times[(pair, "modern", block, operation, KWAY)]
                s = times[
                    (
                        pair,
                        "modern",
                        block,
                        "sequential" if operation == "kway" else "p50",
                        SEQ,
                    )
                ]
                o = times[
                    (
                        pair,
                        "original",
                        block,
                        "sequential" if operation == "kway" else "p50",
                        SEQ,
                    )
                ]
                row.update(
                    {
                        f"modern_kway_b{block}_ns": k,
                        f"modern_sequential_b{block}_ns": s,
                        f"original_sequential_b{block}_ns": o,
                        f"kway_over_modern_sequential_b{block}": k / s,
                        f"kway_over_original_sequential_b{block}": k / o,
                    }
                )
            row.update(
                kway_b2_over_b1=row["modern_kway_b2_ns"] / row["modern_kway_b1_ns"],
                modern_sequential_b2_over_b1=row["modern_sequential_b2_ns"]
                / row["modern_sequential_b1_ns"],
                original_control_b2_over_b1=row["original_sequential_b2_ns"]
                / row["original_sequential_b1_ns"],
                modern_kway_policy_violation=state_metrics[(pair, "modern", KWAY)][
                    "policy_violation"
                ],
            )
            kway_rows.append(row)
        for route in ("direct", SEQ, KWAY):
            t1, t2 = [times[(pair, "modern", b, "cdf", route)] for b in (1, 2)]
            cdf_rows.append(
                {
                    **base,
                    "route": route,
                    "block1_ns_per_call": t1,
                    "block2_ns_per_call": t2,
                    "block2_over_block1": t2 / t1,
                    "policy_violation": state_metrics[(pair, "modern", route)][
                        "policy_violation"
                    ],
                }
            )
    write_csv(results / "paired-timings.csv", paired_rows)
    write_csv(results / "modern-kway.csv", kway_rows)
    write_csv(results / "modern-cdf.csv", cdf_rows)
    write_csv(results / "state-accuracy-storage.csv", list(state_metrics.values()))
    write_csv(results / "tail-accuracy.csv", tails)
    write_csv(results / "policy-evaluations-both-blocks.csv", policy_evaluations)
    for name, rows in (
        ("batch-timings.jsonl", batches),
        ("identity-controls.jsonl", compact_controls),
    ):
        with (results / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    misses = [r for r in state_metrics.values() if r["policy_violation"]]
    audit = {
        "status": "complete",
        "processes": 72,
        "pairs": 18,
        "block_state_observations": 180,
        "unique_states": 90,
        "timing_rows": len(batches),
        "cross_block_state_controls": len(controls),
        "query_control_rows": sum(len(c["query_controls"]) for c in compact_controls),
        "unique_policy_misses": misses,
        "policy_misses_in_both_blocks": sum(r["violation"] for r in policy_evaluations),
        "inherited_original_observations_both_blocks": dict(inherited),
        "source_build_sha256": digest(run / "build.json"),
        "source_metadata_sha256": digest(run / "full/metadata.json"),
        "source_binary_sha256": build["binary_sha256"],
        "selection_sha256": meta["selection_sha256"],
        "recorded_full_seconds_including_reference_and_checks": meta["seconds"],
        "raw_sha256": meta["raw_sha256"],
    }
    (results / "analysis-validation.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if args.independent_review is not None:
        shutil.copy2(
            args.independent_review, results / "independent-native-review.json"
        )
    for source, target in (
        ("build.json", "build.json"),
        ("full/metadata.json", "full-metadata.json"),
        ("smoke/metadata.json", "smoke-metadata.json"),
        ("selection-lock.json", "selection-lock.json"),
    ):
        shutil.copy2(run / source, results / target)
    for name in (
        "run_timing.py",
        "timing_tail.rs",
        "PLAN.md",
        "PROTOCOL.draft.json",
        "test_guards.py",
        "analyze_matched_timing.py",
        "CHECKS.json",
        "preparation-checks.log",
    ):
        shutil.copy2(Path(__file__).parent / name, output / name)
    (output / "FULL_RESULTS_LOCATION.json").write_text(
        json.dumps(
            {
                "run_directory": str(run),
                "raw_files_included": False,
                "native_binary_included": False,
                "raw_manifest": "results/analysis-validation.json",
                "reason": "Numeric timing batches, accuracy summaries, controls and provenance are included; full native states/query arrays and immutable source exports remain at the saved run directory.",
            },
            indent=2,
        )
        + "\n"
    )

    def ratio_pair(pair, operation, route):
        row = next(
            r
            for r in paired_rows
            if r["pair"] == pair and r["operation"] == operation and r["route"] == route
        )
        return (
            f"{row['modern_over_original_b1']:.3f}/{row['modern_over_original_b2']:.3f}"
        )

    lines = [
        "# Timing at training-selected storage/error policies",
        "",
        f"Opt-in modern K2Norm with singleton policy Off at `{runner.MODERN}`, versus the independent default original algorithm at `{runner.ORIGINAL}`. These are personal-repository measurements using one native Rust release binary, the same jemalloc/dependencies, WSL Linux, CPU 0 and one thread. Fresh seed 86028121, n = 1,000,000, normal/ties/separated modes, F32/F64 means. No native code was executed by this analysis.",
        "",
        "All 72 scheduled processes completed: 90 unique states repeated across two counterbalanced blocks, 468 timing rows and 36 unchanged state/leaf/query controls. **Eight unique states missed their selected policy** (16 block observations): the training-selected policy is not a guarantee at n = 1M. This extrapolates beyond the accuracy training/held-out maximum n = 100K.",
        "",
        "For normal f64 at the 2 pp policy, both cores meet the queried rank threshold: modern build takes 0.825/0.868 times original and direct p50 takes 0.789/0.729 times original across the two blocks, while sequential merging takes 1.542/1.681 times original (about 54–68% slower). This policy does not match actual storage; the Python compatibility adapter is not timed.",
        "",
        "## All 18 pairs",
        "",
        "Each timing entry is **modern/original elapsed time, block 1/block 2**; below 1 means less modern time. Each block value is the median of its fixed batches, not a pooled estimate or significance test. Original then modern ran in block 1; reversed core, case and operation order in block 2. Direct and sequential query states are distinct. The original control's and modern core's block 2/block 1 drift are retained for every row in the linked CSV.",
        "",
        "| Pair | Policy | Input | Means | Delta O/M | Build M/O | Sequential merge M/O | Direct p50 M/O | Sequential p50 M/O | Policy misses |",
        "|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for pair, selected in enumerate(build["pairs"]):
        notes = []
        for r in misses:
            if r["pair"] == pair:
                amount = (
                    f"{r['logical_bytes']} B"
                    if selected["policies"][0]["policy"] == "storage"
                    else f"{100 * r['max_rank_error_fraction']:.6f} pp"
                )
                notes.append(
                    f"{r['core']} {'direct' if r['route'] == 'direct' else 'kway' if r['route'] == KWAY else 'seq'} {amount}"
                )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(pair),
                    label(selected["policies"]),
                    selected["kind"],
                    selected["precision"],
                    f"{selected['original_delta']}/{selected['modern_delta']}",
                    ratio_pair(pair, "build", "direct"),
                    ratio_pair(pair, "sequential", SEQ),
                    ratio_pair(pair, "p50", "direct"),
                    ratio_pair(pair, "p50", SEQ),
                    "; ".join(notes) or "none",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Operations, controls and actual accuracy",
        "",
        "[Paired timings](results/paired-timings.csv) contains both absolute ns/operation timings, both M/O ratios and both cores' block 2/block 1 drift for every same-route operation. [Modern k-way](results/modern-kway.csv) includes absolute times and both k-way/modern-sequential and k-way/original-sequential ratios; these compare different merge routes and retain original control drift. [Modern CDF](results/modern-cdf.csv) gives absolute time only for direct/sequential/k-way states; the original API has no CDF comparison. [Integer batch timings](results/batch-timings.jsonl) retain every measured elapsed time and iteration count.",
        "",
        "For a concrete tradeoff, normal f64 at the 2 pp policy uses delta 1200 for both cores: modern build time is 0.825/0.868 times original, while sequential merge costs 1.542/1.681 times original. Modern k-way instead takes 0.993/0.902 ms versus original sequential 1.460/1.390 ms; that is a different route with its own recorded state accuracy. Modern scalar CDF on the direct state takes 1.220/1.219 us per call; no original CDF ratio is available.",
        "",
        "Block drift limits the size of any inferred gain. For normal f64 at 1 KiB, sequential-state p50 has M/O ratios 0.545/0.280, but the unchanged original control slows by 1.679 times between blocks while modern time changes by 0.863 times. Both observations are retained; the larger apparent advantage in block 2 cannot be attributed to a code change between those blocks.",
        "",
        "[State accuracy/storage](results/state-accuracy-storage.csv) includes actual bytes, centroid counts, per-route p50 and maximum queried rank/value errors, CDF errors and all policy misses. [Tail accuracy](results/tail-accuracy.csv) retains six fixed requested/effective quantile points. [Policy evaluations](results/policy-evaluations-both-blocks.csv) preserve both block observations. Matching training policies does not mean equal actual error, equal retained capacity, or equal RSS. Comparisons labelled as satisfying a bound require both states to meet it; misses remain in this primary table.",
        "",
        "Original vector capacity is measured from actual timed states, without compaction; the compatibility Python adapter's shrink-to-fit behavior is not timed. Modern capacity is unknown. Logical object/live-centroid bytes, original retained vector capacity, and modern wire bytes are different quantities. All numerical, leaf, capacity, input and timed-query controls were identical across blocks; this does not eliminate thermal/frequency/scheduler variation.",
        "",
        "Build includes original internal copy/sort and modern slice-to-owned-vector copy. Merge includes reset/ownership clones and excludes prebuilding 40 leaves. Output destruction follows timed build/merge batches; the public one-element modern CDF includes result allocation/destruction. Fixed samples: build has 1 warmup plus 3 single builds; merges have 2 warmups plus 5 batches of 8 reductions; p50 and CDF have 2 warmup batches plus 5 measured batches of 4096 and 4095 calls respectively. No adaptive calibration or retuning.",
        "",
        "## Limits and provenance",
        "",
        "This has two process observations per core/cell on one machine, one fresh seed and three inputs; it is descriptive evidence, not a significance test or world-best result. Timing does not pick deltas. The 0.5/1.0-point error policies are unavailable within the frozen 25–1200 grid; 2 KiB policies were selected but deliberately outside this timing subset. All 24 selection decisions remain in the [selection lock](results/selection-lock.json). Range/skew/reversed shard layouts, other scales/policies/platforms and Python adapter speed are not measured here. Modern gains require opting into that core; this does not establish unchanged performance of the compatibility API or its adapter/compaction changes. Large-n Type 7 references are shared and identity-checked, not independently regenerated by this report.",
        "",
        "[Analysis validation](results/analysis-validation.json), [build provenance](results/build.json), [full execution metadata](results/full-metadata.json), [smoke metadata](results/smoke-metadata.json), and [identity controls](results/identity-controls.jsonl) preserve sources, locks, binary, schedule and raw hashes. Full source exports, native binary and raw query/state arrays remain outside this compact bundle at [saved locations](FULL_RESULTS_LOCATION.json). [README](README.md) labels the preserved historical preparation documents. The full driver duration includes input/reference calculations and checks and is not a library operation timing.",
        "",
    ]
    if args.independent_review is not None:
        lines += [
            "The [independent native review](results/independent-native-review.json) separately verifies the saved full run, including numerical controls and every policy miss.",
            "",
        ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    (output / "README.md").write_text(
        "# Matched-budget timing evidence\n\nRead [RESULTS.md](RESULTS.md) for all 18 paired cells and explicit policy misses. Numeric tables, batches and provenance are in [results](results/analysis-validation.json). [PACKAGE-MANIFEST.json](PACKAGE-MANIFEST.json) hashes the compact bundle.\n\n[PLAN.md](PLAN.md), [PROTOCOL.draft.json](PROTOCOL.draft.json), [CHECKS.json](CHECKS.json) and [preparation-checks.log](preparation-checks.log) are historical preparation records retained unchanged. Their pending/no-execution wording predates the completed build/smoke/full run and is superseded by the result metadata. Frozen hashes must remain unchanged.\n\nReproduce only with the pinned source/accuracy protocol, completed training selection and independent smoke, following the preserved PLAN and run_timing.py phases. Reanalyze a saved run with `python3 analyze_matched_timing.py RUN_DIRECTORY FRESH_OUTPUT_DIRECTORY`; analysis runs no native code. Full native source exports/data are recorded in [FULL_RESULTS_LOCATION.json](FULL_RESULTS_LOCATION.json), not embedded in this bundle.\n"
    )
    manifest = {
        "status": "complete",
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
                "output": str(output),
                "files": len(manifest["files_sha256"]) + 1,
                "bytes": sum(
                    p.stat().st_size for p in output.rglob("*") if p.is_file()
                ),
                "processes": 72,
                "unique_states": 90,
                "unique_policy_misses": len(misses),
                "misses_both_blocks": audit["policy_misses_in_both_blocks"],
                "timing_rows": len(batches),
                "full_run_seconds": meta["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
