#!/usr/bin/env python3
"""Guarded accuracy-only runner. Static plan only unless --execute is supplied."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib

import accuracy_reference as reference
from held_out_plan import DISTRIBUTIONS, HELD_OUT, TRAINING, cases

MODERN = "52f6212008111b62fa07374278b850868b9ea77e"
ORIGINAL = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--phase", choices=["smoke", "training", "held-out"], default="smoke"
    )
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cargo-target", type=Path)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--runtime-budget-seconds", type=float)
    parser.add_argument("--case-timeout-seconds", type=float, default=120)
    parser.add_argument("--build-timeout-seconds", type=float, default=1800)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    planned = cases(args.phase)
    hashes = {
        name: digest(here / name)
        for name in [
            "run_held_out.py",
            "held_out_accuracy.rs",
            "accuracy_reference.py",
            "held_out_plan.py",
        ]
    }
    protocol = {
        "schema": 1,
        "modern_commit": MODERN,
        "original_commit": ORIGINAL,
        "study": "Fixed delta 300 accuracy; no matched-memory selection or speed claim",
        "source_files_sha256": hashes,
        "training_seeds": TRAINING,
        "held_out_seeds": HELD_OUT,
        "distributions": DISTRIBUTIONS,
        "sizes": [32, 1000, 100000],
        "original_delta": 300,
        "modern_delta": 300,
        "scale": "K2Norm",
        "singleton_policy": "Off",
        "mean_precisions": ["f64", "f32"],
        "reverse_shard_order_at_n": 100000,
        "timed_operations": 0,
    }
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "not executed",
                    "phase": args.phase,
                    "datasets": len(planned),
                    "states": sum(c["expected_state_records"] for c in planned),
                    "protocol": protocol,
                    "gates": [
                        "CPU/cache assignment",
                        "exact-rational reference probes",
                        "complete tiny smoke before survey",
                        "matching frozen protocol before held-out",
                    ],
                },
                indent=2,
            )
        )
        return
    if sys.platform != "linux" or not all([args.repo, args.upstream, args.output]):
        parser.error("Execution requires Linux/WSL and --repo --upstream --output")
    if min(args.case_timeout_seconds, args.build_timeout_seconds, args.jobs) <= 0:
        parser.error("Timeouts and job count must be positive")
    budget = (
        args.runtime_budget_seconds
        if args.runtime_budget_seconds is not None
        else (180 if args.phase == "smoke" else 900)
    )
    if budget <= 0:
        parser.error("Runtime budget must be positive")
    if args.phase != "smoke":
        if not args.smoke_evidence:
            parser.error(
                "Survey phases require --smoke-evidence from this exact protocol"
            )
        smoke = json.loads((args.smoke_evidence / "metadata.json").read_text())
        assert smoke["status"] == "complete" and smoke["phase"] == "smoke"
        assert smoke["completed_cases"] == 16 and smoke["record_count"] == 176
        assert (
            smoke["protocol"] == protocol
            and smoke["reference_checks"]["status"] == "pass"
        )
        assert smoke["reference_checks"]["tiny_numeric_cells"] > 0
        assert not smoke["failures"] and not smoke["source_changes"]
        for name, expected in smoke["compressed_case_sha256"].items():
            assert digest(args.smoke_evidence / "cases" / name) == expected, name
    if args.phase == "held-out":
        if not args.protocol_lock:
            parser.error(
                "Held-out execution requires the previously frozen --protocol-lock"
            )
        assert json.loads(args.protocol_lock.read_text()) == protocol
    repo, upstream, output = (
        args.repo.resolve(),
        args.upstream.resolve(),
        args.output.resolve(),
    )
    if repo == upstream:
        parser.error("Original checkout must be independent")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir()
    (output / "logs").mkdir()
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    env.update(RUSTUP_TOOLCHAIN="1.90.0", RAYON_NUM_THREADS="1")
    env["CARGO_TARGET_DIR"] = str(
        (args.cargo_target or output / "cargo-target").resolve()
    )
    metadata = {
        "status": "preparing",
        "phase": args.phase,
        "started_utc": now(),
        "platform": platform.platform(),
        "protocol": protocol,
        "planned_cases": planned,
        "completed_cases": 0,
        "record_count": 0,
        "failures": [],
        "case_results": [],
        "source_changes": [],
        "compressed_case_sha256": {},
        "runtime_budget_seconds": budget,
        "case_timeout_seconds": args.case_timeout_seconds,
        "scope": "No operation timings; wall durations are execution-budget metadata only",
    }

    def save():
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def command(argv, cwd=repo):
        return subprocess.check_output(argv, cwd=cwd, env=env)

    def blob(checkout, commit, name):
        return command(["git", "-C", str(checkout), "show", f"{commit}:{name}"])

    save()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    metadata["manual_oracle"] = reference.verify_manual_oracle()
    for checkout, commit in [(repo, MODERN), (upstream, ORIGINAL)]:
        assert (
            command(["git", "-C", str(checkout), "rev-parse", commit + "^{commit}"])
            .decode()
            .strip()
            == commit
        )
    source = output / "source"
    source.mkdir()
    archive = command(["git", "archive", "--format=tar", MODERN])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(source, filter="data")
    metadata["modern_tree"] = (
        command(["git", "rev-parse", MODERN + "^{tree}"]).decode().strip()
    )
    original = output / "upstream-core"
    (original / "src").mkdir(parents=True)
    provenance = json.loads(
        blob(repo, MODERN, "compat/performance/original-source-provenance.json")
    )
    for name, expected in provenance["normalized_lf_sha256"].items():
        content = blob(upstream, ORIGINAL, "src/" + name).replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected, name
        (original / "src" / name).write_bytes(content)
    original_manifest = tomllib.loads(blob(upstream, ORIGINAL, "Cargo.toml").decode())
    lines = [
        "[package]",
        'name="tdigest_core"',
        f"version={json.dumps(original_manifest['package']['version'])}",
        f"edition={json.dumps(original_manifest['package']['edition'])}",
        "[dependencies]",
    ]
    for name, version in original_manifest["dependencies"].items():
        assert isinstance(version, str)
        lines.append(f"{name}={json.dumps(version)}")
    (original / "Cargo.toml").write_text("\n".join(lines) + "\n")
    project = output / "harness"
    (project / "src").mkdir(parents=True)
    shutil.copyfile(here / "held_out_accuracy.rs", project / "src/main.rs")
    assert digest(project / "src/main.rs") == hashes["held_out_accuracy.rs"]
    (project / "Cargo.toml").write_text(
        '[package]\nname="held-out-accuracy"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\n'
        f'gr-tdigest={{path={json.dumps(str(source))}}}\ntdigest_core={{path={json.dumps(str(original))}}}\nserde_json="1.0.145"\n'
    )
    shutil.copyfile(source / "Cargo.lock", project / "Cargo.lock")
    tracked = (
        list((source / "src").rglob("*.rs"))
        + [source / "Cargo.toml", source / "Cargo.lock"]
        + list((original / "src").glob("*.rs"))
        + [original / "Cargo.toml", project / "Cargo.toml", project / "src/main.rs"]
    )
    metadata["source_sha256"] = {str(p.relative_to(output)): digest(p) for p in tracked}
    metadata["rustc"] = command(["rustc", "-Vv"]).decode().strip()
    metadata["environment"] = {
        k: v
        for k, v in env.items()
        if k.startswith("CARGO_PROFILE_")
        or k
        in [
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RAYON_NUM_THREADS",
            "LD_PRELOAD",
            "MALLOC_CONF",
        ]
    }
    metadata["status"] = "building"
    save()
    with (output / "build.log").open("w") as log:
        subprocess.run(
            ["cargo", "build", "--release", "--offline", "--jobs", str(args.jobs)],
            cwd=project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.build_timeout_seconds,
            check=True,
        )
    binary = output / "held-out-accuracy"
    shutil.copyfile(Path(env["CARGO_TARGET_DIR"]) / "release/held-out-accuracy", binary)
    binary.chmod(0o555)
    metadata["binary_sha256"] = digest(binary)
    metadata["resolved_lock_sha256"] = digest(project / "Cargo.lock")
    metadata["status"] = "reference-gate"
    save()
    # The probe exercises reference helpers only; the tiny smoke additionally
    # verifies every emitted reference against actual post-cast samples.
    probe = subprocess.run(
        [str(binary), "--reference-probe"],
        cwd=output,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.case_timeout_seconds,
        check=True,
    )
    (output / "reference-probe.jsonl").write_text(probe.stdout)
    metadata["reference_probe"] = reference.verify_reference_probe(
        [json.loads(line) for line in probe.stdout.splitlines()]
    )
    metadata["reference_checks"] = {"status": "pass", "tiny_numeric_cells": 0}
    metadata["status"] = "running"
    save()
    start = time.monotonic()
    for index, case in enumerate(planned):
        if time.monotonic() - start >= budget:
            metadata["budget_exhausted"] = True
            break
        assert digest(binary) == metadata["binary_sha256"]
        stem = (
            f"{index:04d}-{case['kind']}-{case['n']}-{case['precision']}-{case['seed']}"
        )
        path = output / "cases" / (stem + ".jsonl.gz")
        errors, row_count, kinds, originals, identities = (
            [],
            0,
            Counter(),
            Counter(),
            set(),
        )
        routes = Counter()
        argv = [
            str(binary),
            case["kind"],
            str(case["n"]),
            str(case["seed"]),
            case["precision"],
            str(case["original_delta"]),
            str(case["modern_delta"]),
            case["merge_order"],
        ]
        case_start = time.monotonic()
        timed_out = threading.Event()
        with (
            (output / "logs" / (stem + ".stderr")).open("w") as stderr,
            gzip.open(path, "wt", encoding="utf-8") as packed,
        ):
            process = subprocess.Popen(
                argv,
                cwd=output,
                env=env,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
            )

            def expire():
                timed_out.set()
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

            timer = threading.Timer(args.case_timeout_seconds, expire)
            timer.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    packed.write(line)
                    try:
                        row = json.loads(line)
                        row_count += 1
                        assert (
                            row["kind"],
                            row["n"],
                            row["precision"],
                            row["seed"],
                        ) == (case["kind"], case["n"], case["precision"], case["seed"])
                        identities.add(row["input_fnv1a64"])
                        kinds[row["core"]] += 1
                        routes[(row["core"], row["route"])] += 1
                        assert (
                            row["original_delta"] == case["original_delta"]
                            and row["modern_delta"] == case["modern_delta"]
                        )
                        if row["route"] == "direct":
                            assert row["shard_sizes"] == []
                        else:
                            assert (
                                len(row["shard_sizes"]) == 40
                                and sum(row["shard_sizes"]) == case["n"]
                            )
                        if row["invariant_failures"]:
                            errors.append(
                                {
                                    "route": row["route"],
                                    "modern_invariants": row["invariant_failures"],
                                }
                            )
                        originals.update(row["inherited_original_observations"])
                        if case["n"] <= 32:
                            metadata["reference_checks"]["tiny_numeric_cells"] += (
                                reference.verify_tiny_row(row)
                            )
                    except (AssertionError, ValueError, KeyError, TypeError) as error:
                        errors.append(
                            {"row": row_count, "reference_or_schema_error": repr(error)}
                        )
                        metadata["reference_checks"]["status"] = "failed"
                return_code = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        expected_routes = Counter({("original", "direct"): 1, ("modern", "direct"): 1})
        orders = (
            ["forward", "reverse"]
            if case["merge_order"] == "both-orders"
            else ["forward"]
        )
        for layout in ["balanced_iid", "balanced_range", "skewed"]:
            for order in orders:
                expected_routes[("original", f"{layout}/{order}/sequential")] = 1
                expected_routes[("modern", f"{layout}/{order}/sequential")] = 1
                expected_routes[("modern", f"{layout}/{order}/kway")] = 1
        if routes != expected_routes:
            errors.append(
                {
                    "route_coverage_mismatch": True,
                    "actual": {str(k): v for k, v in routes.items()},
                }
            )
        if row_count != case["expected_state_records"] or len(identities) != 1:
            errors.append(
                {
                    "record_count": row_count,
                    "expected": case["expected_state_records"],
                    "input_fingerprints": len(identities),
                }
            )
        if return_code != 0 or timed_out.is_set():
            errors.append({"return_code": return_code, "timeout": timed_out.is_set()})
        result = {
            "case": case,
            "records": row_count,
            "core_records": dict(kinds),
            "inherited_original_observations": dict(originals),
            "failures": errors,
            "elapsed_seconds_execution_metadata": time.monotonic() - case_start,
        }
        metadata["case_results"].append(result)
        metadata["record_count"] += row_count
        metadata["completed_cases"] += 1
        metadata["compressed_case_sha256"][path.name] = digest(path)
        if errors:
            metadata["failures"].append(
                {"index": index, "case": case, "errors": errors}
            )
        save()
        print(
            f"{args.phase} case={index + 1}/{len(planned)} records={row_count} failures={len(errors)}",
            flush=True,
        )
    metadata["source_changes"] = [
        name
        for name, expected in metadata["source_sha256"].items()
        if digest(output / name) != expected
    ]
    metadata["finished_utc"] = now()
    metadata["runtime_seconds_execution_metadata"] = time.monotonic() - start
    metadata["status"] = (
        "failed"
        if metadata["failures"] or metadata["source_changes"]
        else "complete"
        if metadata["completed_cases"] == len(planned)
        else "incomplete"
    )
    save()
    if metadata["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
