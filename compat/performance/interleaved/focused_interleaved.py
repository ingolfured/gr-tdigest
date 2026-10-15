#!/usr/bin/env python3
"""Deferred immutable ABBA/BAAB tdigest comparison; no execution without --execute."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import tomllib

VERSIONS = {
    "baseline": "22867bd55739a96a6fab3c8c48b27406ecb32197",
    "candidate": "c5dec35059990b40fe6f036dc83590bf0cdf2a70",
}
ORIGINAL = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"
CASES = [
    ("uniform", 10_000_000, "f64"),
    ("uniform", 10_000_000, "f32"),
    ("sorted", 100_000, "f64"),
    ("sorted", 100_000, "f32"),
]

SMOKE_CASES = [
    ("uniform", 32, "f64"),
    ("uniform", 32, "f32"),
    ("sorted", 32, "f64"),
    ("sorted", 32, "f32"),
]


def run_schedule(smoke: bool) -> list[dict]:
    """Pure schedule generation; no source access, build or benchmark execution."""
    result = []
    if smoke:
        for index, (kind, n, precision) in enumerate(SMOKE_CASES):
            labels = list(VERSIONS) if index % 2 == 0 else list(reversed(VERSIONS))
            for position, label in enumerate(labels):
                result.append(
                    dict(
                        kind=kind,
                        n=n,
                        precision=precision,
                        round=0,
                        position=position,
                        version=label,
                        order="original-first" if index % 2 == 0 else "modern-first",
                    )
                )
        return result
    for round_index in range(2):
        cases = CASES if round_index == 0 else list(reversed(CASES))
        labels = (
            ["baseline", "candidate", "candidate", "baseline"]
            if round_index == 0
            else ["candidate", "baseline", "baseline", "candidate"]
        )
        for kind, n, precision in cases:
            for position, label in enumerate(labels):
                result.append(
                    dict(
                        kind=kind,
                        n=n,
                        precision=precision,
                        round=round_index,
                        position=position,
                        version=label,
                        order="original-first" if round_index == 0 else "modern-first",
                    )
                )
    return result


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Build and time only after the CPU/cache slot is assigned",
    )
    parser.add_argument(
        "--repo", type=Path, help="Repository containing both immutable commits"
    )
    parser.add_argument(
        "--upstream", type=Path, help="Independent original tdigest-rs checkout"
    )
    parser.add_argument(
        "--output", type=Path, help="Fresh output directory; never overwritten"
    )
    parser.add_argument(
        "--cargo-target",
        type=Path,
        help="Optional existing release cache; reserve it before execution",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Exercise eight tiny cases without timing; still requires --execute for builds/runs",
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    planned_runs = run_schedule(args.smoke)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "not executed",
                    "versions": VERSIONS,
                    "mode": "smoke" if args.smoke else "timing",
                    "cases": SMOKE_CASES if args.smoke else CASES,
                    "process_count": len(planned_runs),
                    "planned_runs": planned_runs,
                    "schedule": "Two rounds, ABBA then BAAB; reverse case and within-process operation order in round 2",
                    "timing": "Disabled"
                    if args.smoke
                    else "Existing function: 3 warmups, 15ms calibration, 7 batches targeting 25ms",
                },
                indent=2,
            )
        )
        return
    if sys.platform != "linux":
        parser.error("Use Linux/WSL only")
    if not all([args.repo, args.upstream, args.output]):
        parser.error("--execute requires --repo, --upstream and --output")
    repo, upstream, output = (
        args.repo.resolve(),
        args.upstream.resolve(),
        args.output.resolve(),
    )
    if repo == upstream:
        parser.error("Original checkout must be independent")
    if output.exists() and any(output.iterdir()):
        parser.error("Refusing to reuse nonempty output")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    env["RUSTUP_TOOLCHAIN"] = "1.90.0"
    env["RAYON_NUM_THREADS"] = "1"
    env["CARGO_TARGET_DIR"] = str(
        (args.cargo_target or output / "cargo-target").resolve()
    )

    def command(arguments, cwd=repo):
        return subprocess.check_output(arguments, cwd=cwd, env=env)

    def git_blob(checkout, commit, name):
        return command(["git", "-C", str(checkout), "show", f"{commit}:{name}"])

    baseline_harness = git_blob(
        repo, VERSIONS["baseline"], "compat/performance/harness.rs"
    )
    assert baseline_harness == git_blob(
        repo, VERSIONS["candidate"], "compat/performance/harness.rs"
    )
    # Retain the exact original imports, RNG/distributions, timing and accuracy helpers.
    shared = baseline_harness.decode().split("macro_rules! suite", 1)[0]
    tail = Path(__file__).with_name("focused_interleaved_tail.rs").read_text()
    harness = shared + tail
    (output / "harness.rs").write_text(harness)
    provenance = json.loads(
        git_blob(
            repo,
            VERSIONS["baseline"],
            "compat/performance/original-source-provenance.json",
        )
    )
    original = output / "upstream-core"
    (original / "src").mkdir(parents=True)
    for name, expected in provenance["normalized_lf_sha256"].items():
        data = git_blob(upstream, ORIGINAL, "src/" + name).replace(b"\r\n", b"\n")
        assert sha(data) == expected, name
        (original / "src" / name).write_bytes(data)
    manifest = tomllib.loads(git_blob(upstream, ORIGINAL, "Cargo.toml").decode())
    lines = [
        "[package]",
        'name="tdigest_core"',
        f"version={json.dumps(manifest['package']['version'])}",
        f"edition={json.dumps(manifest['package']['edition'])}",
        "[dependencies]",
    ]
    for name, version in manifest["dependencies"].items():
        assert isinstance(version, str)
        lines.append(f"{name}={json.dumps(version)}")
    (original / "Cargo.toml").write_text("\n".join(lines) + "\n")
    metadata = {
        "status": "building",
        "mode": "smoke" if args.smoke else "timing",
        "planned_runs": planned_runs,
        "started_utc": now(),
        "versions": VERSIONS,
        "original_pin": ORIGINAL,
        "python": sys.version,
        "platform": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        "rustc": command(["rustc", "-Vv"]).decode().strip(),
        "full_matrix_harness_sha256": sha(baseline_harness),
        "focused_harness_sha256": sha(harness.encode()),
        "profile": "cargo release defaults; same process jemalloc and dependencies for both cores",
        "environment": {
            k: env.get(k)
            for k in [
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RUSTC",
                "RAYON_NUM_THREADS",
            ]
        },
        "snapshots": {},
        "binaries": {},
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    binaries = {}
    snapshots = {}
    locks = []
    (output / "bin").mkdir()
    for label, commit in VERSIONS.items():
        assert (
            command(["git", "-C", str(repo), "rev-parse", commit + "^{commit}"])
            .decode()
            .strip()
            == commit
        )
        snapshot = output / "sources" / label
        snapshot.mkdir(parents=True)
        archive = command(["git", "-C", str(repo), "archive", "--format=tar", commit])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source_archive:
            source_archive.extractall(snapshot, filter="data")
        hashes = {
            str(p.relative_to(snapshot)): sha(p.read_bytes())
            for p in (snapshot / "src").rglob("*.rs")
        }
        metadata["snapshots"][label] = {
            "commit": commit,
            "tree": command(["git", "rev-parse", commit + "^{tree}"]).decode().strip(),
            "source_sha256": hashes,
            "cargo_manifest_sha256": sha((snapshot / "Cargo.toml").read_bytes()),
            "cargo_lock_sha256": sha((snapshot / "Cargo.lock").read_bytes()),
        }
        snapshots[label] = snapshot
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        project = output / "harness-crates" / label
        (project / "src").mkdir(parents=True)
        (project / "src/main.rs").write_text(harness)
        (project / "Cargo.toml").write_text(
            '[package]\nname="focused-interleaved-digest"\nversion="0.0.0"\nedition="2021"\n'
            "[workspace]\n[dependencies]\n"
            f"gr-tdigest={{path={json.dumps(str(snapshot))}}}\n"
            f'tdigest_core={{path={json.dumps(str(original))}}}\nserde_json="1.0.145"\n'
        )
        shutil.copyfile(snapshot / "Cargo.lock", project / "Cargo.lock")
        with (output / f"build-{label}.log").open("w") as log:
            subprocess.run(
                ["cargo", "build", "--release", "--offline", "--jobs", str(args.jobs)],
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        binary = output / "bin" / label
        shutil.copyfile(
            Path(env["CARGO_TARGET_DIR"]) / "release/focused-interleaved-digest", binary
        )
        binary.chmod(0o555)
        binaries[label] = binary
        metadata["binaries"][label] = {
            "path": str(binary),
            "sha256": sha(binary.read_bytes()),
        }
        locks.append((project / "Cargo.lock").read_bytes())
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    assert locks[0] == locks[1], "Resolved dependency locks differ"
    assert (
        metadata["snapshots"]["baseline"]["cargo_manifest_sha256"]
        == metadata["snapshots"]["candidate"]["cargo_manifest_sha256"]
    )
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    metadata.update(
        status="smoke" if args.smoke else "timing",
        affinity_cpu=cpu,
        resolved_lock_sha256=sha(locks[0]),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records, expectations, mismatches = [], {}, []
    with (output / "results.jsonl").open("w") as results:
        for planned in planned_runs:
            kind, n, precision = planned["kind"], planned["n"], planned["precision"]
            label, round_index = planned["version"], planned["round"]
            position, order = planned["position"], planned["order"]
            case = (kind, n, precision)
            assert (
                sha(binaries[label].read_bytes())
                == metadata["binaries"][label]["sha256"]
            )
            started = now()
            start = time.monotonic()
            result = subprocess.run(
                [
                    str(binaries[label]),
                    kind,
                    str(n),
                    precision,
                    order,
                    "smoke" if args.smoke else "timing",
                ],
                cwd=output,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(result.stdout)
            assert (row["kind"], row["n"], row["precision"]) == case
            assert row["mode"] == metadata["mode"]
            if args.smoke:
                assert all(
                    row[op] is None
                    for op in (
                        "original_build",
                        "modern_build",
                        "original_p50",
                        "modern_p50",
                    )
                )
            row.update(
                version=label,
                source_commit=VERSIONS[label],
                round=round_index,
                schedule_position=position,
                started_utc=started,
                elapsed_seconds=time.monotonic() - start,
                loadavg_after=Path("/proc/loadavg").read_text().strip(),
            )
            if case not in expectations:
                expectations[case] = row["state"]
            elif row["state"] != expectations[case]:
                mismatches.append(
                    dict(case=case, round=round_index, position=position, version=label)
                )
            records.append(row)
            results.write(json.dumps(row, allow_nan=False) + "\n")
            results.flush()
            print(
                f"{metadata['mode']} round={round_index + 1} {kind} n={n} {precision} {label} complete",
                flush=True,
            )
    comparisons = []
    for round_index in [] if args.smoke else range(2):
        for kind, n, precision in CASES:
            group = [
                r
                for r in records
                if (r["kind"], r["n"], r["precision"], r["round"])
                == (kind, n, precision, round_index)
            ]
            for operation in [
                "original_build",
                "modern_build",
                "original_p50",
                "modern_p50",
            ]:
                medians = {
                    label: [
                        r[operation]["median_ns"]
                        for r in group
                        if r["version"] == label
                    ]
                    for label in VERSIONS
                }
                comparisons.append(
                    {
                        "round": round_index,
                        "kind": kind,
                        "n": n,
                        "precision": precision,
                        "operation": operation,
                        "run_medians_ns": medians,
                        "candidate_over_baseline": statistics.median(
                            medians["candidate"]
                        )
                        / statistics.median(medians["baseline"]),
                    }
                )
    changed = {
        label: [
            name
            for name, expected in metadata["snapshots"][label]["source_sha256"].items()
            if sha((snapshots[label] / name).read_bytes()) != expected
        ]
        for label in VERSIONS
    }
    metadata.update(
        status="failed" if mismatches or any(changed.values()) else "complete",
        completed_processes=len(records),
        finished_utc=now(),
        source_changes=changed,
        numerical_state_mismatches=mismatches,
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Smoke correctness/API checks only; no performance measurements or claims"
                if args.smoke
                else "Interleaved block medians; descriptive ratios, no causal or statistical significance claim",
                "numerical_state_mismatches": mismatches,
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )
    if mismatches or any(changed.values()):
        raise SystemExit(
            "Invalid comparison: source or numerical state changed; inspect retained records"
        )


if __name__ == "__main__":
    main()
