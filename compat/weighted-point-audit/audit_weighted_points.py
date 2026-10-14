#!/usr/bin/env python3
"""Bounded exact weighted-point audit; ordinary failures exit with status 1."""

import argparse
import hashlib
import importlib
import platform
import subprocess
import sys
import collections
import fractions
import json
import math
import random
import struct
import time
from pathlib import Path

import gr_tdigest as td

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=202610140639)
parser.add_argument(
    "--trials", type=int, default=12, help="positive count of random point sets"
)
parser.add_argument("--scales", default="1e-300,1e-46,0.1,1,16777216,1e9,1e39,1e300")
parser.add_argument("--output", type=Path, default=Path("weighted-point-audit.json"))
parser.add_argument(
    "--source-dir",
    type=Path,
    help="optional checkout for source hashes; not used for imports",
)
parser.add_argument(
    "--include-extremes",
    action="store_true",
    help="report known limitations separately; never part of the passing gate",
)
args = parser.parse_args()
if args.trials < 1:
    parser.error("--trials must be positive")
try:
    configured_scales = [float(value) for value in args.scales.split(",")]
except ValueError:
    parser.error("--scales must be a comma-separated list of positive finite floats")
if not configured_scales or any(
    not math.isfinite(scale) or scale <= 0 for scale in configured_scales
):
    parser.error("--scales must contain positive finite floats")
SEED = args.seed
rng = random.Random(SEED)
start = time.perf_counter()
failures = []
checks = collections.Counter()


def record(kind, data):
    checks["failures_" + kind] += 1
    if len(failures) < 25 or kind.endswith("exception"):
        failures.append({"kind": kind, **data})


def reference(values, weights):
    points = collections.defaultdict(fractions.Fraction)
    for x, w in zip(values, weights):
        points[x] += fractions.Fraction(w)
    return sorted(points.items())


def verify(digest, values, weights, precision, route, trial, scale):
    points = reference(values, weights)
    total = sum((w for _, w in points), fractions.Fraction())
    context = dict(
        precision=precision,
        route=route,
        trial=trial,
        scale=scale,
        values=values,
        weights=weights,
        means=list(digest.means),
        stored_weights=list(digest.weights),
    )
    actual_points = list(zip(digest.means.tolist(), digest.weights.tolist()))
    if [x for x, _ in actual_points] != [x for x, _ in points]:
        record("atomic_mean_drift", context)
    for (x, got), (_, expected) in zip(actual_points, points):
        if not math.isclose(got, float(expected), rel_tol=2e-14, abs_tol=0):
            record(
                "mass", {**context, "x": x, "actual": got, "expected": float(expected)}
            )
    probes = [points[0][0] - 1, points[-1][0] + 1]
    probes.extend(x for x, _ in points)
    probes.extend((a[0] + b[0]) / 2 for a, b in zip(points, points[1:]))
    actual_cdf = list(digest.cdf(probes))
    for x, actual in zip(probes, actual_cdf):
        left = sum((w for v, w in points if v < x), fractions.Fraction())
        equal = sum((w for v, w in points if v == x), fractions.Fraction())
        expected = float((left + equal / 2) / total)
        checks["cdf"] += 1
        if not math.isfinite(actual) or abs(actual - expected) > 5e-14:
            record(
                "cdf", {**context, "probe": x, "actual": actual, "expected": expected}
            )
    qs = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    prefix = fractions.Fraction()
    for _, w in points:
        qs.extend([float(prefix / total), float((prefix + w / 2) / total)])
        prefix += w
    actual_q = list(digest.quantile(qs))
    for q, x in zip(qs, actual_q):
        checks["quantile_rank"] += 1
        left = float(sum((w for v, w in points if v < x), fractions.Fraction()) / total)
        right = float(
            sum((w for v, w in points if v <= x), fractions.Fraction()) / total
        )
        if not math.isfinite(x) or not left - 5e-14 <= q <= right + 5e-14:
            record(
                "quantile_rank",
                {**context, "q": q, "actual": x, "valid_rank_interval": [left, right]},
            )
    if all(float(w).is_integer() for w in weights) and total <= 2**53:
        count = struct.unpack_from("<d", digest.to_bytes(), 20)[0]
        checks["integer_mass"] += 1
        if count != float(total) or sum(digest.weights) != float(total):
            record(
                "integer_mass", {**context, "count": count, "expected": float(total)}
            )
    checks["digests"] += 1


scales = configured_scales
for trial in range(args.trials):
    values = [float(rng.randrange(-5, 6)) for _ in range(12)]
    base_weights = [float(rng.randrange(1, 10)) for _ in values]
    for scale in scales:
        weights = [w * scale for w in base_weights]
        for precision in ["f32", "f64"]:
            for route in ["original", "reverse", "shuffle"]:
                indices = list(range(len(values)))
                if route == "reverse":
                    indices.reverse()
                elif route == "shuffle":
                    rng.shuffle(indices)
                try:
                    digest = td.TDigest.from_means_weights(
                        [values[i] for i in indices],
                        [weights[i] for i in indices],
                        precision=precision,
                        max_size=1000,
                    )
                    verify(digest, values, weights, precision, route, trial, scale)
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    record(
                        "build_exception",
                        dict(
                            values=values,
                            weights=weights,
                            precision=precision,
                            route=route,
                            trial=trial,
                            scale=scale,
                            error_type=type(error).__name__,
                            error=str(error),
                        ),
                    )
            sequential = td.TDigest.from_array([], precision=precision, max_size=1000)
            try:
                for x, w in zip(values, weights):
                    sequential.add_weighted(x, w)
                verify(
                    sequential,
                    values,
                    weights,
                    precision,
                    "sequential_add",
                    trial,
                    scale,
                )
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                record(
                    "sequential_exception",
                    dict(
                        values=values,
                        weights=weights,
                        precision=precision,
                        route="sequential_add",
                        trial=trial,
                        scale=scale,
                        error_type=type(error).__name__,
                        error=str(error),
                    ),
                )
            if trial in [0, 5] and scale in [1e-46, 0.1, 1.0, 1e9]:
                shard_values = [values[i % len(values)] for i in range(40)]
                shard_weights = [weights[i % len(weights)] for i in range(40)]
                shards = [
                    td.TDigest.from_means_weights([x], [w], precision=precision)
                    for x, w in zip(shard_values, shard_weights)
                ]
                try:
                    merged = td.TDigest.merge_all(shards)
                    verify(
                        merged,
                        shard_values,
                        shard_weights,
                        precision,
                        "merge40",
                        trial,
                        scale,
                    )
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    record(
                        "merge_exception",
                        dict(
                            values=shard_values,
                            weights=shard_weights,
                            precision=precision,
                            route="merge40",
                            trial=trial,
                            scale=scale,
                            error_type=type(error).__name__,
                            error=str(error),
                        ),
                    )
                rng.shuffle(shards)
                try:
                    verify(
                        td.TDigest.merge_all(shards),
                        shard_values,
                        shard_weights,
                        precision,
                        "merge40_shuffled",
                        trial,
                        scale,
                    )
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    record(
                        "merge_exception",
                        dict(
                            values=shard_values,
                            weights=shard_weights,
                            precision=precision,
                            route="merge40_shuffled",
                            trial=trial,
                            scale=scale,
                            error_type=type(error).__name__,
                            error=str(error),
                        ),
                    )

outside = []
for precision in ["f32", "f64"] if args.include_extremes else []:
    for label, values, weights in [
        ("one_min_subnormal", [0.0], [math.ulp(0.0)]),
        ("two_min_subnormal", [0.0, 1.0], [math.ulp(0.0)] * 2),
        ("aggregate_overflow_distinct", [0.0, 1.0], [1e308, 1e308]),
        ("aggregate_overflow_duplicate", [0.0, 0.0], [1e308, 1e308]),
    ]:
        case = dict(label=label, precision=precision, values=values, weights=weights)
        try:
            digest = td.TDigest.from_means_weights(values, weights, precision=precision)
            case.update(
                means=digest.means.tolist(),
                stored_weights=digest.weights.tolist(),
                cdf=list(digest.cdf([0.0, 0.5, 1.0])),
                quantiles=list(digest.quantile([0.0, 0.25, 0.5, 0.6, 0.75, 1.0])),
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            case.update(error_type=type(error).__name__, error=str(error))
        outside.append(case)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(source_dir):
    if source_dir is None:
        return None
    root = source_dir.resolve()
    files = sorted((root / "src" / "tdigest").glob("*.rs"))
    files.extend(
        root / name for name in ["src/py.rs", "bindings/python/gr_tdigest/__init__.py"]
    )
    if not files or not all(path.is_file() for path in files):
        raise ValueError("--source-dir must identify a gr-tdigest checkout")
    record = {
        "directory": str(root),
        "sha256": {str(path.relative_to(root)): sha256_file(path) for path in files},
    }
    try:
        record["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        record["working_tree_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        )
        record["tracked_diff_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        ).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        record["commit"] = None
    return record


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


elapsed = time.perf_counter() - start
native = importlib.import_module("gr_tdigest._gr_tdigest")
failure_count = sum(
    count for name, count in checks.items() if name.startswith("failures_")
)
report = dict(
    status="PASS" if failure_count == 0 else "FAIL",
    seed=SEED,
    controls=dict(
        trials=args.trials,
        scales=scales,
        precisions=["f32", "f64"],
        max_size=1000,
        values_per_trial=12,
        integer_value_range=[-5, 5],
        integer_weight_range=[1, 9],
        merge_fan_in=40,
        merge_trials=[0, 5],
        merge_scales=[1e-46, 0.1, 1.0, 1e9],
        rank_absolute_tolerance=5e-14,
        weight_relative_tolerance=2e-14,
    ),
    seconds=elapsed,
    checks=dict(checks),
    ordinary_failure_count=failure_count,
    failure_details=failures,
    known_extreme_probes=outside,
    known_extreme_probes_are_part_of_gate=False,
    environment=dict(
        python=sys.version,
        platform=platform.platform(),
        package_version=td.__version__,
        package_path=td.__file__,
        native_path=native.__file__,
        native_sha256=sha256_file(native.__file__),
        audit_script_sha256=sha256_file(__file__),
    ),
    source=source_provenance(args.source_dir),
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        dict(
            status=report["status"],
            output=str(args.output),
            seed=SEED,
            seconds=elapsed,
            checks=dict(checks),
            ordinary_failure_count=failure_count,
            known_extreme_probe_count=len(outside),
        )
    )
)
raise SystemExit(1 if failure_count else 0)
