"""Scratch selected-budget timing executor. Dry plan unless --execute; Linux only.

No delta overrides. Requires a completed corrected-source accuracy study and an
explicit selection hash. Build/smoke/full are separate, non-overwriting phases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import tomllib

MODERN = "ff73cfab31d6368d203c1a361b8efaaea29d735b"
ORIGINAL = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"
ACCURACY_PROTOCOL = "9c2a455e689802eb2dbbe7764764238b794fa5975e2517d5e3c1355fed66904a"
POLICIES = [("storage", 1024), ("storage", 4096), ("rank", 0.02)]
KINDS = ["normal", "ties", "separated_modes"]
PRECISIONS = ["f32", "f64"]
GRID = [25, 50, 100, 150, 200, 300, 600, 1200]
SEED = 86028121
LIMIT_BYTES = 64 * 1024**2
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_files(root):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def implementation():
    names = [
        "PLAN.md",
        "PROTOCOL.draft.json",
        "timing_tail.rs",
        "run_timing.py",
        "test_guards.py",
    ]
    return {name: digest(HERE / name) for name in names}


def selections(lock):
    require(
        lock["status"] == "locked_from_training_only",
        "Selection not frozen from training",
    )
    require(lock["design"]["modern_commit"] == MODERN, "Wrong modern source")
    require(lock["design"]["original_commit"] == ORIGINAL, "Wrong original source")
    require(lock["design"]["delta_grid"] == GRID, "Unexpected delta grid")
    lookups = {}
    for kind, field, key in [
        ("storage", "storage_selections", "logical_cap_bytes"),
        ("rank", "illustrative_rank_error_selections", "rank_error_threshold"),
    ]:
        rows = lock[field]
        require(len(rows) == 12, "Incomplete selection table")
        table = {(r["core"], r["precision"], r[key]): r for r in rows}
        limits = [1024, 2048, 4096] if kind == "storage" else [0.005, 0.01, 0.02]
        require(
            set(table)
            == {
                (c, p, v)
                for c in ["original", "modern"]
                for p in PRECISIONS
                for v in limits
            },
            "Selection coverage changed",
        )
        for row in rows:
            require(
                (row["status"] == "selected" and row["delta"] in GRID)
                or (
                    row["status"] == "unavailable_within_grid" and row["delta"] is None
                ),
                "Invalid selected status/delta",
            )
        lookups[kind] = table
    grouped, skipped = {}, []
    for policy, threshold in POLICIES:
        for kind in KINDS:
            for precision in PRECISIONS:
                chosen = {
                    core: lookups[policy][core, precision, threshold]
                    for core in ["original", "modern"]
                }
                label = {"policy": policy, "threshold": threshold}
                if any(r["delta"] is None for r in chosen.values()):
                    skipped.append(
                        {
                            **label,
                            "kind": kind,
                            "precision": precision,
                            "reason": "At least one core unavailable; no substitute timing",
                            "selections": chosen,
                        }
                    )
                    continue
                key = (
                    kind,
                    precision,
                    chosen["original"]["delta"],
                    chosen["modern"]["delta"],
                )
                if key not in grouped:
                    grouped[key] = {
                        "kind": kind,
                        "precision": precision,
                        "seed": SEED,
                        "original_delta": key[2],
                        "modern_delta": key[3],
                        "policies": [],
                    }
                grouped[key]["policies"].append(label)
    return list(grouped.values()), skipped


def schedule(pairs, smoke=False):
    runs = []
    for block in [1] if smoke else [1, 2]:
        for index in range(len(pairs)) if block == 1 else reversed(range(len(pairs))):
            for core in (
                ["original", "modern"] if block == 1 else ["modern", "original"]
            ):
                runs.append(
                    {
                        **pairs[index],
                        "pair_index": index,
                        "core": core,
                        "block": block,
                        "n": 32 if smoke else 1_000_000,
                        "smoke": smoke,
                    }
                )
    return runs


class Stop(RuntimeError):
    pass


def run_child(command, stdout, stderr, deadline, remaining_bytes, *, per_process=30.0):
    """Linux process-group deadline and streamed byte cap, retaining partial files."""
    started = time.monotonic()
    remaining = min(per_process, deadline - started)
    if remaining <= 0 or remaining_bytes <= 0:
        raise Stop("Global time/output budget exhausted before launch")
    child = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    count = 0
    try:
        with (
            Path(stdout).open("wb") as out,
            Path(stderr).open("wb") as err,
            selectors.DefaultSelector() as selector,
        ):
            selector.register(child.stdout, selectors.EVENT_READ, out)
            selector.register(child.stderr, selectors.EVENT_READ, err)
            while selector.get_map():
                left = min(deadline, started + per_process) - time.monotonic()
                if left <= 0:
                    raise Stop("Process/global deadline reached")
                for event, _ in selector.select(min(left, 0.05)):
                    data = os.read(event.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(event.fileobj)
                        continue
                    allowed = max(0, remaining_bytes - count)
                    event.data.write(data[:allowed])
                    event.data.flush()
                    count += min(len(data), allowed)
                    if len(data) > allowed or count >= remaining_bytes:
                        raise Stop("Raw output byte cap reached")
            try:
                code = child.wait(
                    timeout=max(
                        0.001, min(deadline, started + per_process) - time.monotonic()
                    )
                )
            except subprocess.TimeoutExpired as error:
                raise Stop(
                    "Process/global deadline reached after pipes closed"
                ) from error
            require(code == 0, f"Native/child exit {code}; raw output preserved")
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        child.stdout.close()
        child.stderr.close()
    return {
        "seconds": time.monotonic() - started,
        "bytes": count,
        "exit_code": child.returncode,
    }


def verify_accuracy(args):
    protocol_path = args.protocol_dir / "execution-protocol.prepared.json"
    require(
        digest(protocol_path) == ACCURACY_PROTOCOL,
        "Corrected accuracy protocol changed",
    )
    protocol = read(protocol_path)
    for name, expected in protocol["source_files_sha256"].items():
        require(
            Path(name).name == name and digest(args.protocol_dir / name) == expected,
            "Frozen accuracy helper changed: " + name,
        )
    require(
        digest(args.selection_lock) == args.selection_sha256, "Selection lock changed"
    )
    lock = read(args.selection_lock)
    pairs, skipped = selections(lock)
    require(
        lock["protocol"] == protocol,
        "Selection protocol differs from approved corrected protocol",
    )
    sys.path.insert(0, str(args.protocol_dir))
    spec = importlib.util.spec_from_file_location(
        "accuracy_driver", args.protocol_dir / "run_storage_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(
        module.execution_protocol(args.protocol_dir) == protocol,
        "Execution helper protocol mismatch",
    )
    ledger = read(args.accuracy_study / "ledger.json")
    provenance = module.verify_build(args.accuracy_study, ledger, protocol)
    phase_metadata = {}
    for phase, cases, records in [
        ("smoke", 128, 1408),
        ("training", 1152, 16128),
        ("held-out", None, None),
    ]:
        require(
            digest(args.accuracy_study / phase / "metadata.json")
            == ledger["phases"][phase]["metadata_sha256"],
            "Accuracy phase metadata differs from its ledger checksum: " + phase,
        )
        value = module.verify_complete_phase(
            args.accuracy_study / phase, phase, protocol, cases, records
        )
        require(
            value["source_sha256"] == provenance["source_sha256"],
            "Incomplete phase source manifest",
        )
        phase_metadata[phase] = value
    training = phase_metadata["training"]
    module.validate_selection_header(
        lock,
        protocol,
        digest(args.accuracy_study / "training/metadata.json"),
        digest(args.accuracy_study / "training/protocol.json"),
        training["compressed_case_sha256"],
    )
    held = phase_metadata["held-out"]
    require(
        held["selection_lock_sha256"] == args.selection_sha256,
        "Held-out used different settings",
    )
    require(
        read(args.accuracy_study / "held-out/recomputed-selection-lock.json") == lock,
        "Held-out training-only selection recheck differs",
    )
    expected, _ = module.held_out_cases(lock)
    require(
        held["completed_cases"] == len(expected)
        and held["record_count"] == sum(c["expected_state_records"] for c in expected),
        "Incomplete held-out policy coverage",
    )
    return protocol, provenance, pairs, skipped, module


def verify_frozen(args):
    require(
        digest(args.output / "build.json")
        == (args.output / "build.sha256").read_text().strip(),
        "Sealed build provenance changed",
    )
    meta = read(args.output / "build.json")
    require(meta["status"] == "complete", "Timing build incomplete")
    require(
        meta["implementation_sha256"] == implementation(),
        "Timing implementation changed after freeze",
    )
    require(
        digest(args.selection_lock)
        == meta["selection_sha256"]
        == args.selection_sha256,
        "Selected settings changed",
    )
    require(
        digest(args.output / "selection-lock.json") == args.selection_sha256,
        "Retained settings changed",
    )
    require(
        digest(args.output / "timing") == meta["binary_sha256"],
        "Timing executable changed",
    )
    require(
        source_files(args.output / "exports") == meta["export_sha256"],
        "Complete immutable export manifest changed",
    )
    require(
        digest(args.output / "harness/Cargo.lock") == meta["resolved_lock_sha256"],
        "Timing Cargo.lock changed",
    )
    require(
        digest(args.output / "harness/Cargo.toml") == meta["harness_manifest_sha256"],
        "Timing manifest changed",
    )
    require(
        digest(args.output / "harness/src/main.rs") == meta["harness_sha256"],
        "Actual compiled harness changed",
    )
    return meta


def build(args):
    require(not args.output.exists(), "Fresh output directory required")
    require(
        not args.output.is_relative_to(args.accuracy_study)
        and not args.output.is_relative_to(args.protocol_dir),
        "Timing output must be outside preserved accuracy evidence/protocol",
    )
    require(
        not args.cargo_target.is_relative_to(args.accuracy_study)
        and not args.cargo_target.is_relative_to(args.protocol_dir),
        "Cargo cache must be outside preserved accuracy evidence/protocol",
    )
    protocol, provenance, pairs, skipped, _ = verify_accuracy(args)
    args.output.mkdir(parents=True)
    meta = {
        "status": "preparing",
        "started_utc": now(),
        "modern_commit": MODERN,
        "original_commit": ORIGINAL,
        "accuracy_protocol_sha256": ACCURACY_PROTOCOL,
        "accuracy_provenance_sha256": digest(
            args.accuracy_study / "build/provenance.json"
        ),
        "selection_sha256": args.selection_sha256,
        "implementation_sha256": implementation(),
        "pairs": pairs,
        "skipped": skipped,
    }
    save(args.output / "build.json", meta)
    try:
        shutil.copyfile(args.selection_lock, args.output / "selection-lock.json")
        save(args.output / "accuracy-protocol.json", protocol)
        exports = args.output / "exports"
        exports.mkdir()
        previous = args.accuracy_study / "build"
        for name in ["source", "upstream-core"]:
            shutil.copytree(previous / name, exports / name)
            require(
                source_files(previous / name) == source_files(exports / name),
                "Immutable export copy changed",
            )
        shutil.copyfile(
            previous / "harness/Cargo.lock", exports / "accuracy-resolved.Cargo.lock"
        )
        shutil.copyfile(
            previous / "provenance.json", exports / "accuracy-build-provenance.json"
        )
        shutil.copyfile(
            args.protocol_dir / "held_out_accuracy.rs", exports / "held_out_accuracy.rs"
        )
        reference = (exports / "held_out_accuracy.rs").read_text()
        require(
            reference.count("macro_rules! precision_suite") == 1,
            "Reference helper split marker changed",
        )
        prefix = reference.split("macro_rules! precision_suite", 1)[0]
        project = args.output / "harness"
        (project / "src").mkdir(parents=True)
        (project / "src/main.rs").write_text(
            prefix + (HERE / "timing_tail.rs").read_text()
        )
        # Keep the same harness package identity/dependencies, changing only path locations
        # and source. Its existing resolved dependency lock can therefore stay --locked.
        manifest = (previous / "harness/Cargo.toml").read_text()
        for name in ["source", "upstream-core"]:
            old, new = json.dumps(str(previous / name)), json.dumps(str(exports / name))
            require(manifest.count(old) == 1, "Unexpected reference dependency path")
            manifest = manifest.replace(old, new)
        parsed = tomllib.loads(manifest)
        require(
            parsed["package"]["name"] == "held-out-accuracy",
            "Unexpected harness identity",
        )
        require(
            parsed["dependencies"]["tdigest_core"]["package"]
            == "tdigest-original-reference",
            "Independent original package alias lost",
        )
        (project / "Cargo.toml").write_text(manifest)
        shutil.copyfile(previous / "harness/Cargo.lock", project / "Cargo.lock")
        before_lock = digest(project / "Cargo.lock")
        env = os.environ.copy()
        require(
            not any(k.startswith("CARGO_PROFILE_") for k in env),
            "Unreviewed Cargo profile override",
        )
        require(not env.get("CARGO_BUILD_TARGET"), "Unreviewed Cargo target override")
        env.update(
            RUSTUP_TOOLCHAIN="1.90.0",
            RAYON_NUM_THREADS="1",
            CARGO_TARGET_DIR=str(args.cargo_target),
        )
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
        for key in [
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTC",
            "LD_PRELOAD",
            "MALLOC_CONF",
        ]:
            require(not env.get(key), "Unreviewed compiler/allocator override: " + key)
        meta.update(
            status="building",
            export_sha256=source_files(exports),
            harness_sha256=digest(project / "src/main.rs"),
            harness_manifest_sha256=digest(project / "Cargo.toml"),
            reference_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
            rustc=subprocess.check_output(["rustc", "-Vv"], env=env, text=True).strip(),
        )
        save(args.output / "build.json", meta)
        print(
            "Building the source-bound timing executable; build.log retains compiler output",
            flush=True,
        )
        with (args.output / "build.log").open("wb") as log:
            subprocess.run(
                ["cargo", "build", "--release", "--locked", "--offline", "--jobs", "2"],
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        require(
            digest(project / "Cargo.lock") == before_lock,
            "Locked dependency graph changed",
        )
        require(
            source_files(exports) == meta["export_sha256"],
            "Source exports changed while building",
        )
        shutil.copyfile(
            args.cargo_target / "release/held-out-accuracy", args.output / "timing"
        )
        (args.output / "timing").chmod(0o555)
        meta.update(
            status="complete",
            binary_sha256=digest(args.output / "timing"),
            resolved_lock_sha256=before_lock,
        )
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        meta["finished_utc"] = now()
        save(args.output / "build.json", meta)
    (args.output / "build.sha256").write_text(digest(args.output / "build.json") + "\n")


def validate_result(row, case, reference=None):
    for key in [
        "core",
        "kind",
        "n",
        "seed",
        "precision",
        "original_delta",
        "modern_delta",
        "block",
        "smoke",
    ]:
        require(row[key] == case[key], "Native case identity mismatch: " + key)
    require(
        len(row["shard_sizes"]) == 40 and sum(row["shard_sizes"]) == case["n"],
        "Wrong shard coverage",
    )
    expected = ["direct", "balanced_iid/forward/sequential"] + (
        ["balanced_iid/forward/kway"] if case["core"] == "modern" else []
    )
    require(
        [s["route"] for s in row["states"]] == expected, "Missing/extra state routes"
    )
    evaluations = []
    for state in row["states"]:
        for key in [
            "core",
            "kind",
            "n",
            "seed",
            "precision",
            "original_delta",
            "modern_delta",
        ]:
            require(state[key] == case[key], "State identity mismatch: " + key)
        require(
            state["input_fnv1a64"] == row["input_fnv1a64"],
            "State input fingerprint differs",
        )
        require(not state["invariant_failures"], "Modern invariant failure")
        if reference is not None:
            reference.verify_tiny_row(state)
        metrics = state["state"]
        require(
            metrics["logical_retained_bytes"]
            == metrics["object_bytes"]
            + metrics["centroids"] * metrics["centroid_bytes"],
            "Logical-byte formula mismatch",
        )
        require(len(metrics["quantile"]["rows"]) == 1003, "Query coverage changed")
        ranks = [q[4] for q in metrics["quantile"]["rows"]]
        finite_ranks = all(x is not None and math.isfinite(x) for x in ranks)
        require(
            case["core"] == "original" or finite_ranks, "Nonfinite modern queried rank"
        )
        for policy in case["policies"]:
            value = (
                metrics["logical_retained_bytes"]
                if policy["policy"] == "storage"
                else (max(ranks) if finite_ranks else None)
            )
            evaluations.append(
                {
                    **policy,
                    "core": case["core"],
                    "route": state["route"],
                    "observed": value,
                    "excess": max(0, value - policy["threshold"])
                    if value is not None
                    else None,
                    "violation": value is None or value > policy["threshold"],
                    "nonfinite_rank_observation": not finite_ranks,
                }
            )
    expected_ops = {
        (op, route)
        for route in expected
        for op in (["p50", "cdf"] if case["core"] == "modern" else ["p50"])
    }
    expected_ops |= {
        ("build", "direct"),
        ("sequential", "balanced_iid/forward/sequential"),
    }
    if case["core"] == "modern":
        expected_ops.add(("kway", "balanced_iid/forward/kway"))
    require(
        {(t["operation"], t["route"]) for t in row["timings"]} == expected_ops
        and len(row["timings"]) == len(expected_ops),
        "Timing coverage changed",
    )
    for timing in row["timings"]:
        samples = timing["ns_per_operation"]
        count = 1 if case["smoke"] else (3 if timing["operation"] == "build" else 5)
        require(
            len(samples) == count and all(math.isfinite(s) and s >= 0 for s in samples),
            "Invalid timing batches",
        )
        iterations = (
            (16 if case["smoke"] else 4096)
            if timing["operation"] == "p50"
            else (
                (15 if case["smoke"] else 4095)
                if timing["operation"] == "cdf"
                else (1 if case["smoke"] or timing["operation"] == "build" else 8)
            )
        )
        require(
            timing["iterations_per_batch"] == iterations,
            "Timing iteration count changed",
        )
        require(
            len(timing["batch_elapsed_ns"]) == count
            and all(
                isinstance(ns, int) and ns >= 0 and sample == ns / iterations
                for ns, sample in zip(timing["batch_elapsed_ns"], samples)
            ),
            "Raw batch elapsed times differ from normalized timings",
        )
        if timing["operation"] in ["p50", "cdf"]:
            require(
                len(timing["checksum_bits"]) == count
                and all(
                    s == timing["expected_checksum_bits"]
                    for s in timing["checksum_bits"]
                ),
                "Timed query checksum differs from untimed expectation",
            )
            probes = 1 if timing["operation"] == "p50" else 3
            require(
                len(timing["query_input_bits"])
                == len(timing["query_value_bits"])
                == probes,
                "Timed query probes changed",
            )
    return evaluations


def execute(args, smoke):
    require(
        not os.environ.get("LD_PRELOAD") and not os.environ.get("MALLOC_CONF"),
        "Unreviewed runtime allocator override",
    )
    meta = verify_frozen(args)
    _, _, _, _, guard = verify_accuracy(args)
    phase = "smoke" if smoke else "full"
    destination = args.output / phase
    require(not destination.exists(), "Never overwrite/resume execution evidence")
    if not smoke:
        require(
            digest(args.output / "smoke/metadata.json")
            == (args.output / "smoke/metadata.sha256").read_text().strip(),
            "Timing smoke metadata changed after completion",
        )
        prior = read(args.output / "smoke/metadata.json")
        require(
            prior["status"] in ["complete", "complete_unavailable"]
            and prior["binary_sha256"] == meta["binary_sha256"]
            and prior["selection_sha256"] == args.selection_sha256,
            "Matching successful timing smoke required",
        )
        require(
            len(prior["completed_runs"]) == len(schedule(meta["pairs"], True)),
            "Timing smoke coverage incomplete",
        )
        for name, expected in prior["raw_sha256"].items():
            require(
                digest(args.output / "smoke" / name) == expected,
                "Smoke raw output changed",
            )
    require(0 in os.sched_getaffinity(0), "CPU0 unavailable; no fallback")
    os.sched_setaffinity(0, {0})
    require(os.sched_getaffinity(0) == {0}, "CPU affinity failed")
    for name in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]:
        os.environ[name] = "1"
    destination.mkdir()
    runs = schedule(meta["pairs"], smoke)
    record = {
        "status": "running",
        "started_utc": now(),
        "phase": phase,
        "binary_sha256": meta["binary_sha256"],
        "selection_sha256": args.selection_sha256,
        "planned_runs": runs,
        "completed_runs": [],
        "raw_sha256": {},
        "failures": [],
        "skipped": meta["skipped"],
        "affinity": [0],
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        "start_load": os.getloadavg(),
        "bytes": 0,
    }
    started = time.monotonic()
    deadline = started + (60 if smoke else 600)
    controls = {}
    seen_block = None
    save(destination / "metadata.json", record)
    try:
        for index, case in enumerate(runs):
            if case["block"] != seen_block:
                verify_frozen(args)
                seen_block = case["block"]
            label = f"{index:03d}-b{case['block']}-p{case['pair_index']}-{case['core']}"
            stdout, stderr = (
                destination / (label + ".json"),
                destination / (label + ".log"),
            )
            command = [
                str(args.output / "timing"),
                case["core"],
                case["kind"],
                case["precision"],
                str(case["n"]),
                str(SEED),
                str(case["original_delta"]),
                str(case["modern_delta"]),
                str(case["block"]),
                str(smoke).lower(),
            ]
            record["current_run"] = {
                "case": case,
                "command": command,
                "stdout": stdout.name,
                "stderr": stderr.name,
            }
            save(destination / "metadata.json", record)
            print(
                f"{phase}: starting {index + 1}/{len(runs)}, block {case['block']}, pair {case['pair_index']}, {case['core']}",
                flush=True,
            )
            observation = run_child(
                command, stdout, stderr, deadline, LIMIT_BYTES - record["bytes"]
            )
            record["bytes"] += observation["bytes"]
            row = read(stdout)
            evaluations = validate_result(row, case, guard.reference if smoke else None)
            identity = {
                k: row[k]
                for k in ["input_fnv1a64", "leaf_states", "states", "shard_sizes"]
            }
            identity["query_controls"] = sorted(
                [
                    {
                        k: t[k]
                        for k in [
                            "operation",
                            "route",
                            "checksum_bits",
                            "expected_checksum_bits",
                            "query_input_bits",
                            "query_value_bits",
                        ]
                    }
                    for t in row["timings"]
                    if t["operation"] in ["p50", "cdf"]
                ],
                key=lambda t: (t["operation"], t["route"]),
            )
            key = (case["pair_index"], case["core"])
            if key in controls:
                require(
                    identity == controls[key],
                    "Same-core bits/state/metrics changed across blocks",
                )
            else:
                controls[key] = identity
            input_key = (case["pair_index"], "input")
            require(
                input_key not in controls
                or controls[input_key] == row["input_fnv1a64"],
                "Paired inputs differ",
            )
            controls[input_key] = row["input_fnv1a64"]
            record["completed_runs"].append(
                {
                    "case": case,
                    "stdout": stdout.name,
                    "stderr": stderr.name,
                    "command": command,
                    "execution": observation,
                    "policy_evaluations": evaluations,
                }
            )
            for path in [stdout, stderr]:
                record["raw_sha256"][path.name] = digest(path)
            save(destination / "metadata.json", record)
        if time.monotonic() > deadline:
            raise Stop("Global wall budget reached during final validation")
        verify_frozen(args)
        if time.monotonic() > deadline:
            raise Stop("Global wall budget reached during source checks")
        record["status"] = "complete" if runs else "complete_unavailable"
    except Stop as error:
        record.update(status="incomplete_budget", error=repr(error))
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        record["failures"].append(repr(error))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["end_load"] = os.getloadavg()
        record["finished_utc"] = now()
        for path in destination.iterdir():
            if path.is_file() and path.name != "metadata.json":
                record["raw_sha256"][path.name] = digest(path)
        record["bytes"] = sum(
            p.stat().st_size
            for p in destination.iterdir()
            if p.is_file() and p.name != "metadata.json"
        )
        save(destination / "metadata.json", record)
        (destination / "metadata.sha256").write_text(
            digest(destination / "metadata.json") + "\n"
        )
    require(
        record["status"].startswith("complete"), "Timing incomplete; evidence retained"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-slot-released", action="store_true")
    parser.add_argument("--phase", choices=["build", "smoke", "full"], default="smoke")
    for name in [
        "protocol-dir",
        "accuracy-study",
        "selection-lock",
        "output",
        "cargo-target",
    ]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "not executed",
                    "source": MODERN,
                    "accuracy_protocol_sha256": ACCURACY_PROTOCOL,
                    "max_pairs": 18,
                    "max_full_processes": 72,
                    "global_seconds": 600,
                    "selection_hash": "required at runtime",
                    "implementation_sha256": implementation(),
                },
                indent=2,
            )
        )
        return
    require(
        args.cpu_slot_released and sys.platform == "linux",
        "Explicit CPU allocation and Linux required",
    )
    require(
        all(
            getattr(args, n)
            for n in [
                "protocol_dir",
                "accuracy_study",
                "selection_lock",
                "output",
                "cargo_target",
                "selection_sha256",
            ]
        ),
        "All source/study/selection/output/cache arguments required",
    )
    require(
        len(args.selection_sha256) == 64
        and all(c in "0123456789abcdef" for c in args.selection_sha256),
        "Exact lowercase selection SHA256 required",
    )
    for name in [
        "protocol_dir",
        "accuracy_study",
        "selection_lock",
        "output",
        "cargo_target",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    if args.phase == "build":
        build(args)
    else:
        execute(args, args.phase == "smoke")


if __name__ == "__main__":
    main()
