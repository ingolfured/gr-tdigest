#!/usr/bin/env python3
"""Guarded storage/error-policy study. Dry plan unless --execute is supplied."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib

import accuracy_reference as reference
import storage_budget_plan as plan

POLICY_SHA = "7f9c1877ca9c08b8718f1d034bc058e5acec08ec194635a182f92ff368facede"
REFERENCE_PACKAGE = "tdigest-original-reference"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_policy(here):
    require(digest(here / "policy-lock.json") == POLICY_SHA, "Policy lock changed")
    policy = read(here / "policy-lock.json")
    require(policy["design"] == plan.design(), "Policy design changed")
    for name, expected in policy["source_files_sha256"].items():
        require(Path(name).name == name, "Invalid policy source name")
        require(
            digest(here / name) == expected, f"Frozen policy source changed: {name}"
        )
    return policy


def execution_protocol(here):
    policy = validate_policy(here)
    return {
        "schema": "storage-budget-execution-v1",
        "policy_lock_sha256": POLICY_SHA,
        "independent_reference_package": REFERENCE_PACKAGE,
        "design": plan.design(),
        "source_files_sha256": {
            **policy["source_files_sha256"],
            "run_storage_budget.py": digest(here / "run_storage_budget.py"),
        },
        "scope": "No operation timing; fixed training policies; logical bytes are not RSS/capacity",
    }


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, phase, prior_seconds, prior_bytes, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        design = plan.design()
        self.seconds = min(
            design["phase_runtime_seconds"][phase],
            design["global_dataset_runtime_seconds"] - prior_seconds,
        )
        self.bytes = min(
            design["phase_compressed_output_bytes"][phase],
            design["global_compressed_output_bytes"] - prior_bytes,
        )

    def elapsed(self):
        return self.clock() - self.started

    def remaining(self):
        return self.seconds - self.elapsed()

    def check(self, compressed_bytes):
        if self.remaining() <= 0:
            raise BudgetStop("Dataset runtime budget reached")
        if compressed_bytes >= self.bytes:
            raise BudgetStop("Compressed raw-output budget reached")


def source_manifest(build):
    paths = [
        path
        for directory in ["source", "upstream-core"]
        for path in (build / directory).rglob("*")
        if path.is_file()
    ]
    paths += [build / "harness/Cargo.toml", build / "harness/src/main.rs"]
    return {str(path.relative_to(build)): digest(path) for path in sorted(paths)}


def verify_build(study, ledger, protocol):
    build = study / "build"
    require(
        digest(build / "provenance.json") == ledger["build_provenance_sha256"],
        "Build provenance changed",
    )
    provenance = read(build / "provenance.json")
    require(
        provenance["protocol"] == protocol and provenance["status"] == "complete",
        "Build protocol/status mismatch",
    )
    require(
        source_manifest(build) == provenance["source_sha256"],
        "Immutable source export or complete source manifest changed",
    )
    require(
        digest(build / "held-out-accuracy") == provenance["binary_sha256"],
        "Executable changed",
    )
    require(
        digest(build / "harness/Cargo.lock") == provenance["resolved_lock_sha256"],
        "Resolved Cargo.lock changed",
    )
    return provenance


def build_once(study, repo, upstream, target, here, protocol):
    build = study / "build"
    build.mkdir()
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    env.update(RUSTUP_TOOLCHAIN="1.90.0", RAYON_NUM_THREADS="1")
    env["CARGO_TARGET_DIR"] = str(target)
    provenance = {"status": "preparing", "started_utc": now(), "protocol": protocol}
    save(build / "provenance.json", provenance)

    def command(argv):
        return subprocess.check_output(argv, cwd=repo, env=env)

    def blob(checkout, commit, name):
        return command(["git", "-C", str(checkout), "show", f"{commit}:{name}"])

    try:
        for checkout, commit in [(repo, plan.MODERN), (upstream, plan.ORIGINAL)]:
            require(
                command(["git", "-C", str(checkout), "rev-parse", commit + "^{commit}"])
                .decode()
                .strip()
                == commit,
                "Pinned Git object unavailable",
            )
        source = build / "source"
        source.mkdir()
        archive = command(
            ["git", "-C", str(repo), "archive", "--format=tar", plan.MODERN]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        provenance["modern_tree"] = (
            command(["git", "-C", str(repo), "rev-parse", plan.MODERN + "^{tree}"])
            .decode()
            .strip()
        )
        original = build / "upstream-core"
        (original / "src").mkdir(parents=True)
        original_pin = read(
            source / "compat/performance/original-source-provenance.json"
        )
        for name, expected in original_pin["normalized_lf_sha256"].items():
            content = blob(upstream, plan.ORIGINAL, "src/" + name).replace(
                b"\r\n", b"\n"
            )
            require(
                hashlib.sha256(content).hexdigest() == expected,
                "Independent original source mismatch",
            )
            (original / "src" / name).write_bytes(content)
        cargo = tomllib.loads(blob(upstream, plan.ORIGINAL, "Cargo.toml").decode())
        lines = [
            "[package]",
            f"name={json.dumps(REFERENCE_PACKAGE)}",
            f"version={json.dumps(cargo['package']['version'])}",
            f"edition={json.dumps(cargo['package']['edition'])}",
            "[dependencies]",
        ]
        for name, version in cargo["dependencies"].items():
            require(
                isinstance(version, str), "Unsupported original dependency declaration"
            )
            lines.append(f"{name}={json.dumps(version)}")
        (original / "Cargo.toml").write_text("\n".join(lines) + "\n")
        project = build / "harness"
        (project / "src").mkdir(parents=True)
        shutil.copy2(here / "held_out_accuracy.rs", project / "src/main.rs")
        (project / "Cargo.toml").write_text(
            '[package]\nname="held-out-accuracy"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\n'
            + f'gr-tdigest={{path={json.dumps(str(source))}}}\ntdigest_core={{package={json.dumps(REFERENCE_PACKAGE)},path={json.dumps(str(original))}}}\nserde_json="1.0.145"\n'
        )
        shutil.copy2(source / "Cargo.lock", project / "Cargo.lock")
        provenance["source_sha256"] = source_manifest(build)
        provenance["rustc"] = command(["rustc", "-Vv"]).decode().strip()
        provenance["environment"] = {
            name: value
            for name, value in env.items()
            if name.startswith("CARGO_PROFILE_")
            or name
            in [
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RAYON_NUM_THREADS",
                "LD_PRELOAD",
                "MALLOC_CONF",
            ]
        }
        provenance["status"] = "building"
        save(build / "provenance.json", provenance)
        with (build / "build.log").open("w") as log:
            subprocess.run(
                ["cargo", "build", "--release", "--offline", "--jobs", "2"],
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=plan.design()["build_timeout_seconds"],
                check=True,
            )
        binary = build / "held-out-accuracy"
        shutil.copy2(target / "release/held-out-accuracy", binary)
        binary.chmod(0o555)
        require(
            source_manifest(build) == provenance["source_sha256"],
            "Source changed during build",
        )
        provenance.update(
            status="complete",
            finished_utc=now(),
            binary_sha256=digest(binary),
            resolved_lock_sha256=digest(project / "Cargo.lock"),
        )
        save(build / "provenance.json", provenance)
    except BaseException as error:
        provenance.update(status="failed", error=repr(error), finished_utc=now())
        save(build / "provenance.json", provenance)
        raise
    return provenance


def verify_complete_phase(
    directory, phase, protocol, expected_cases=None, expected_records=None
):
    metadata = read(directory / "metadata.json")
    require(
        metadata["status"] == "complete" and metadata["phase"] == phase,
        "Prior phase is not complete",
    )
    require(
        metadata["protocol"] == protocol
        and read(directory / "protocol.json") == protocol,
        "Prior protocol changed",
    )
    require(
        digest(directory / "policy-lock.json") == POLICY_SHA, "Prior policy changed"
    )
    require(
        not metadata["failures"] and not metadata["source_changes"],
        "Prior phase failed",
    )
    require(
        metadata["reference_checks"]["status"] == "pass"
        and metadata["reference_checks"]["tiny_numeric_cells"] > 0,
        "Prior reference checks missing",
    )
    if expected_cases is not None:
        require(
            metadata["completed_cases"] == expected_cases
            and metadata["record_count"] == expected_records,
            "Prior coverage mismatch",
        )
    require(
        digest(directory / "held-out-accuracy") == metadata["binary_sha256"],
        "Prior executable changed",
    )
    require(
        digest(directory / "harness/Cargo.lock") == metadata["resolved_lock_sha256"],
        "Prior dependency lock changed",
    )
    for name, expected in metadata["source_sha256"].items():
        require(digest(directory / name) == expected, f"Prior source changed: {name}")
    actual_files = {path.name for path in (directory / "cases").glob("*.jsonl.gz")}
    require(
        actual_files == set(metadata["compressed_case_sha256"]),
        "Prior raw file coverage changed",
    )
    for name, expected in metadata["compressed_case_sha256"].items():
        require(
            digest(directory / "cases" / name) == expected,
            f"Prior raw data changed: {name}",
        )
    return metadata


def held_out_cases(selection):
    storage = {
        (row["core"], row["precision"], row["logical_cap_bytes"]): row
        for row in selection["storage_selections"]
    }
    rank = {
        (row["core"], row["precision"], row["rank_error_threshold"]): row
        for row in selection["illustrative_rank_error_selections"]
    }
    require(
        len(storage) == 12 and len(rank) == 12, "Incomplete selection policy coverage"
    )
    cases, skipped = [], []
    for policy, thresholds, choices in [
        ("storage", plan.BUDGETS, storage),
        ("rank", plan.RANK_BUDGETS, rank),
    ]:
        for threshold in thresholds:
            for precision in plan.PRECISIONS:
                chosen = {
                    core: choices[(core, precision, threshold)]["delta"]
                    for core in ["original", "modern"]
                }
                require(
                    all(
                        value is None or value in plan.DELTAS
                        for value in chosen.values()
                    ),
                    "Selected delta outside frozen grid",
                )
                if all(value is None for value in chosen.values()):
                    skipped.append(
                        {
                            "policy": policy,
                            "threshold": threshold,
                            "precision": precision,
                            "reason": "both cores unavailable within grid",
                        }
                    )
                    continue
                for seed in plan.HELD_OUT:
                    for kind in plan.DISTRIBUTIONS:
                        for n in plan.SIZES:
                            cases.append(
                                {
                                    "policy": policy,
                                    "threshold": threshold,
                                    "core_policy_status": {
                                        core: "selected"
                                        if delta is not None
                                        else "diagnostic_only_unavailable"
                                        for core, delta in chosen.items()
                                    },
                                    "original_delta": chosen["original"]
                                    or min(plan.DELTAS),
                                    "modern_delta": chosen["modern"]
                                    or min(plan.DELTAS),
                                    "seed": seed,
                                    "kind": kind,
                                    "n": n,
                                    "precision": precision,
                                    "expected_state_records": 20 if n == 100000 else 11,
                                }
                            )
    return cases, skipped


def validate_selection_header(
    selection,
    protocol,
    training_metadata_hash,
    training_protocol_hash,
    training_raw_hashes,
):
    require(
        selection["protocol"] == protocol
        and selection["protocol_sha256"] == training_protocol_hash,
        "Selection protocol mismatch",
    )
    require(
        selection["training_metadata_sha256"] == training_metadata_hash,
        "Selection training metadata changed",
    )
    require(
        selection["training_raw_sha256"] == training_raw_hashes,
        "Selection training raw hashes changed",
    )


def expected_routes(case):
    return Counter(
        {
            (core, route): 1
            for core in ["original", "modern"]
            for route in plan.routes(case["n"], core)
        }
    )


def validate_row(row, case):
    for name in ["kind", "n", "seed", "precision", "original_delta", "modern_delta"]:
        require(row[name] == case[name], f"Row identity mismatch: {name}")
    require(
        (row["core"], row["route"]) in expected_routes(case), "Unexpected core/route"
    )
    if row["route"] == "direct":
        require(row["shard_sizes"] == [], "Unexpected direct shards")
    else:
        require(
            len(row["shard_sizes"]) == 40 and sum(row["shard_sizes"]) == case["n"],
            "Shard coverage mismatch",
        )
    checks = reference.verify_tiny_row(row) if case["n"] <= 32 else 0
    return checks


def evaluate_policy(row, case):
    if "policy" not in case:
        return None
    core = row["core"]
    state = row["state"]
    result = {
        "core": core,
        "route": row["route"],
        "status": case["core_policy_status"][core],
        "logical_bytes": state["logical_retained_bytes"],
        "policy": case["policy"],
        "threshold": case["threshold"],
    }
    if result["status"] != "selected":
        result["violation"] = None
    elif case["policy"] == "storage":
        result["violation"] = result["logical_bytes"] > case["threshold"]
        result["excess_bytes"] = max(0, result["logical_bytes"] - case["threshold"])
    else:
        errors = [q[4] for q in state["quantile"]["rows"]]
        maximum = (
            max(errors)
            if all(value is not None and math.isfinite(value) for value in errors)
            else None
        )
        result["max_queried_rank_error"] = maximum
        result["violation"] = maximum is None or maximum > case["threshold"]
    return result


def stream_case(
    argv,
    path,
    stderr_path,
    timeout,
    remaining_bytes,
    consume,
    line_limit,
    cwd=None,
    env=None,
):
    """Bound one child and preserve raw bytes even when parsing or budgets fail."""
    timed_out = threading.Event()
    result = {"records": 0, "failures": [], "stopped_reason": None}
    with path.open("xb") as raw, stderr_path.open("w") as stderr:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed:
            child = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=stderr
            )

            def expire():
                timed_out.set()
                if child.poll() is None:
                    child.kill()

            timer = threading.Timer(timeout, expire)
            timer.start()
            try:
                while True:
                    line = child.stdout.readline(line_limit + 1)
                    if not line:
                        break
                    packed.write(line)
                    packed.flush()
                    raw.flush()
                    if len(line) > line_limit:
                        result["failures"].append(
                            {"oversized_line_prefix_saved": len(line)}
                        )
                        result["stopped_reason"] = "JSON line limit exceeded"
                        break
                    if raw.tell() >= remaining_bytes:
                        result["stopped_reason"] = (
                            "Compressed raw-output budget reached"
                        )
                        break
                    try:
                        row = json.loads(line)
                        result["records"] += 1
                        consume(row)
                    except BudgetStop as error:
                        result["stopped_reason"] = str(error)
                        break
                    except (
                        AssertionError,
                        ValueError,
                        KeyError,
                        TypeError,
                        IndexError,
                    ) as error:
                        result["failures"].append(
                            {
                                "row": result["records"],
                                "schema_reference_error": repr(error),
                            }
                        )
                if result["stopped_reason"] and child.poll() is None:
                    child.kill()
                result["return_code"] = child.wait()
            finally:
                timer.cancel()
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()
    result["timeout"] = timed_out.is_set()
    if result["return_code"] != 0 or result["timeout"]:
        result["failures"].append(
            {"return_code": result["return_code"], "timeout": result["timeout"]}
        )
    result["compressed_bytes"] = path.stat().st_size
    result["sha256"] = digest(path)
    return result


def phase_snapshot(study, output, provenance, here, protocol):
    for name in ["source", "upstream-core", "harness"]:
        shutil.copytree(study / "build" / name, output / name)
    shutil.copy2(study / "build/held-out-accuracy", output / "held-out-accuracy")
    shutil.copy2(here / "policy-lock.json", output / "policy-lock.json")
    save(output / "protocol.json", protocol)
    require(
        source_manifest(output) == provenance["source_sha256"],
        "Phase source snapshot mismatch",
    )


def run_phase(args, here, protocol):
    study = args.study.resolve()
    if args.phase == "smoke":
        require(
            not study.exists() or not any(study.iterdir()),
            "Smoke requires a fresh study directory",
        )
        require(
            args.repo and args.upstream and args.cargo_target,
            "Smoke requires --repo, --upstream and --cargo-target",
        )
        repo, upstream = args.repo.resolve(), args.upstream.resolve()
        require(repo != upstream, "Original checkout must be independent")
        study.mkdir(parents=True, exist_ok=True)
        save(study / "protocol.json", protocol)
        (study / "driver-snapshot").mkdir()
        for name in protocol["source_files_sha256"]:
            shutil.copy2(here / name, study / "driver-snapshot" / name)
        shutil.copy2(
            here / "policy-lock.json", study / "driver-snapshot/policy-lock.json"
        )
        build_once(study, repo, upstream, args.cargo_target.resolve(), here, protocol)
        ledger = {
            "protocol_sha256": digest(study / "protocol.json"),
            "build_provenance_sha256": digest(study / "build/provenance.json"),
            "phases": {},
        }
        save(study / "ledger.json", ledger)
    else:
        ledger = read(study / "ledger.json")
        require(
            read(study / "protocol.json") == protocol
            and digest(study / "protocol.json") == ledger["protocol_sha256"],
            "Study execution protocol changed",
        )
    provenance = verify_build(study, ledger, protocol)
    for phase, entry in ledger["phases"].items():
        require(
            digest(study / phase / "metadata.json") == entry["metadata_sha256"],
            "Prior phase metadata changed",
        )
    prior_seconds = sum(entry["runtime_seconds"] for entry in ledger["phases"].values())
    prior_bytes = sum(
        path.stat().st_size
        for phase in ["smoke", "training", "held-out"]
        for path in (study / phase / "cases").glob("*.jsonl.gz")
    )
    output = study / args.phase
    require(
        not output.exists(),
        "Phase output exists; stopped/complete phases cannot be overwritten or resumed",
    )
    output.mkdir()
    (output / "cases").mkdir()
    (output / "logs").mkdir()
    phase_snapshot(study, output, provenance, here, protocol)
    budget = Budget(args.phase, prior_seconds, prior_bytes)
    metadata = {
        "status": "preparing",
        "phase": args.phase,
        "started_utc": now(),
        "protocol": protocol,
        "build_provenance_sha256": ledger["build_provenance_sha256"],
        "planned_cases": [],
        "completed_cases": 0,
        "record_count": 0,
        "case_results": [],
        "failures": [],
        "source_changes": [],
        "compressed_case_sha256": {},
        "compressed_bytes": 0,
        "source_sha256": provenance["source_sha256"],
        "binary_sha256": provenance["binary_sha256"],
        "resolved_lock_sha256": provenance["resolved_lock_sha256"],
        "reference_checks": {"status": "pending", "tiny_numeric_cells": 0},
        "global_prior_seconds": prior_seconds,
        "global_prior_compressed_bytes": prior_bytes,
        "effective_phase_runtime_seconds": budget.seconds,
        "effective_phase_compressed_bytes": budget.bytes,
    }
    selection_path, selection_hash = None, None

    def checkpoint():
        metadata["runtime_seconds_execution_metadata"] = budget.elapsed()
        save(output / "metadata.json", metadata)
        ledger["phases"][args.phase] = {
            "status": metadata["status"],
            "runtime_seconds": budget.elapsed(),
            "compressed_bytes": metadata["compressed_bytes"],
            "metadata_sha256": digest(output / "metadata.json"),
        }
        save(study / "ledger.json", ledger)

    checkpoint()
    try:
        budget.check(0)
        if args.phase != "smoke":
            prior = verify_complete_phase(study / "smoke", "smoke", protocol, 128, 1408)
            require(
                prior["source_sha256"] == provenance["source_sha256"],
                "Smoke source manifest incomplete/changed",
            )
        if args.phase == "held-out":
            training = verify_complete_phase(
                study / "training", "training", protocol, 1152, 16128
            )
            require(
                training["source_sha256"] == provenance["source_sha256"],
                "Training source manifest incomplete/changed",
            )
            require(
                args.selection_lock is not None,
                "Held-out requires an existing --selection-lock",
            )
            selection_path = args.selection_lock.resolve()
            selection_hash = digest(selection_path)
            selection = read(selection_path)
            validate_selection_header(
                selection,
                protocol,
                digest(study / "training/metadata.json"),
                digest(study / "training/protocol.json"),
                training["compressed_case_sha256"],
            )
            recheck = output / "recomputed-selection-lock.json"
            budget.check(0)
            with (output / "selection-recheck.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(here / "select_storage_budget.py"),
                        "--training",
                        str(study / "training"),
                        "--output",
                        str(recheck),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.001, budget.remaining()),
                    check=True,
                )
            require(
                read(recheck) == selection,
                "Selection differs from training-only recomputation",
            )
            require(
                digest(selection_path) == selection_hash,
                "Selection changed during preflight",
            )
            shutil.copy2(selection_path, output / "selection-lock.json")
            metadata["selection_lock_sha256"] = selection_hash
            cases, skipped = held_out_cases(selection)
            metadata["unavailable_policy_cells"] = skipped
        else:
            cases = list(plan.training_cases(smoke=args.phase == "smoke"))
        metadata["planned_cases"] = cases
        metadata["manual_oracle"] = reference.verify_manual_oracle()
        budget.check(0)
        binary = output / "held-out-accuracy"
        probe = subprocess.run(
            [str(binary), "--reference-probe"],
            cwd=output,
            capture_output=True,
            text=True,
            timeout=min(120, max(0.001, budget.remaining())),
            check=True,
        )
        (output / "reference-probe.jsonl").write_text(probe.stdout)
        metadata["reference_probe"] = reference.verify_reference_probe(
            [json.loads(line) for line in probe.stdout.splitlines()]
        )
        metadata["reference_checks"]["status"] = "pass"
        metadata["status"] = "running"
        checkpoint()
        for index, case in enumerate(cases):
            budget.check(metadata["compressed_bytes"])
            require(
                digest(binary) == metadata["binary_sha256"],
                "Executable changed mid-phase",
            )
            stem = f"{index:04d}-{case.get('policy', 'grid')}-{case.get('threshold', 'all')}-d{case['original_delta']}-{case['modern_delta']}-{case['kind']}-{case['n']}-{case['precision']}-{case['seed']}"
            path = output / "cases" / (stem + ".jsonl.gz")
            identities, routes, observations, evaluations, invariant_failures = (
                set(),
                Counter(),
                Counter(),
                [],
                [],
            )

            def consume(row):
                budget.check(metadata["compressed_bytes"] + path.stat().st_size)
                checks = validate_row(row, case)
                metadata["reference_checks"]["tiny_numeric_cells"] += checks
                identities.add(row["input_fnv1a64"])
                routes[(row["core"], row["route"])] += 1
                observations.update(row["inherited_original_observations"])
                if row["invariant_failures"]:
                    invariant_failures.append(
                        {
                            "core": row["core"],
                            "route": row["route"],
                            "failures": row["invariant_failures"],
                        }
                    )
                evaluation = evaluate_policy(row, case)
                if evaluation is not None:
                    evaluations.append(evaluation)

            remaining = budget.remaining()
            argv = [
                str(binary),
                case["kind"],
                str(case["n"]),
                str(case["seed"]),
                case["precision"],
                str(case["original_delta"]),
                str(case["modern_delta"]),
                "both-orders" if case["n"] == 100000 else "forward",
            ]
            result = stream_case(
                argv,
                path,
                output / "logs" / (stem + ".stderr"),
                min(120, remaining),
                budget.bytes - metadata["compressed_bytes"],
                consume,
                plan.design()["max_json_line_bytes"],
                cwd=output,
                env={**os.environ, "RAYON_NUM_THREADS": "1"},
            )
            if result["timeout"] and remaining <= 120:
                result["stopped_reason"] = "Dataset runtime budget reached"
            if (
                routes != expected_routes(case)
                or result["records"] != case["expected_state_records"]
                or len(identities) != 1
            ):
                result["failures"].append(
                    {
                        "coverage_mismatch": True,
                        "routes": {str(key): count for key, count in routes.items()},
                        "input_fingerprint_count": len(identities),
                    }
                )
            result["failures"].extend(invariant_failures)
            if any("schema_reference_error" in error for error in result["failures"]):
                metadata["reference_checks"]["status"] = "failed"
            result.update(
                case=case,
                inherited_original_observations=dict(observations),
                policy_evaluations=evaluations,
            )
            metadata["case_results"].append(result)
            metadata["completed_cases"] += 1
            metadata["record_count"] += result["records"]
            metadata["compressed_bytes"] += result["compressed_bytes"]
            metadata["compressed_case_sha256"][path.name] = result["sha256"]
            if result["failures"]:
                metadata["failures"].append(
                    {"case_index": index, "errors": result["failures"]}
                )
            checkpoint()
            print(
                f"{args.phase} {index + 1}/{len(cases)} states={result['records']} failures={len(result['failures'])}",
                flush=True,
            )
            if result["stopped_reason"]:
                raise BudgetStop(result["stopped_reason"])
        metadata["status"] = "failed" if metadata["failures"] else "complete"
    except BudgetStop as error:
        metadata.update(status="incomplete", stopped_reason=str(error))
    except BaseException as error:
        metadata.update(status="failed", exception=repr(error))
        metadata["failures"].append({"phase_exception": repr(error)})
    finally:
        try:
            require(
                execution_protocol(here) == protocol,
                "Driver or policy changed during phase",
            )
            verify_build(study, ledger, protocol)
            require(
                source_manifest(output) == provenance["source_sha256"],
                "Phase source changed",
            )
            require(
                digest(output / "held-out-accuracy") == metadata["binary_sha256"],
                "Phase binary changed",
            )
            require(
                digest(output / "harness/Cargo.lock")
                == metadata["resolved_lock_sha256"],
                "Phase Cargo.lock changed",
            )
            if selection_path is not None:
                require(
                    digest(selection_path) == selection_hash,
                    "Selection lock changed during held-out",
                )
                require(
                    digest(output / "selection-lock.json") == selection_hash,
                    "Retained selection snapshot changed",
                )
        except (ValueError, OSError, KeyError) as error:
            metadata["source_changes"].append(repr(error))
            metadata["status"] = "failed"
        metadata["finished_utc"] = now()
        metadata["compressed_case_sha256"] = {
            path.name: digest(path) for path in (output / "cases").glob("*.jsonl.gz")
        }
        metadata["compressed_bytes"] = sum(
            path.stat().st_size for path in (output / "cases").glob("*.jsonl.gz")
        )
        checkpoint()
    return 0 if metadata["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--phase", choices=["smoke", "training", "held-out"], default="smoke"
    )
    parser.add_argument("--study", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--cargo-target", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    protocol = execution_protocol(here)
    if not args.execute:
        print(
            json.dumps(
                {"status": "not executed", "phase": args.phase, "protocol": protocol},
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux" or not args.study:
        parser.error("Execution requires Linux/WSL and --study")
    return run_phase(args, here, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
