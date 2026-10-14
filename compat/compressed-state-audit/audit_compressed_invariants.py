#!/usr/bin/env python3
"""Deterministic compressed-state invariants; this is not a benchmark or rank oracle."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

import gr_tdigest as td


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def adjacent_one(offset, precision):
    if precision == "f32":
        return struct.unpack("<f", struct.pack("<I", 0x3F800000 + offset))[0]
    return struct.unpack("<d", struct.pack("<Q", 0x3FF0000000000000 + offset))[0]


def datasets(seed, precision, trial):
    rng = random.Random(seed + trial * 1009)
    cast = float32 if precision == "f32" else float
    return {
        "duplicate_heavy": [float(rng.randrange(-3, 4)) for _ in range(40)],
        "adjacent_positive": [adjacent_one(i, precision) for i in range(40)],
        "adjacent_signed_duplicates": [
            (-1 if i % 3 == 0 else 1) * adjacent_one(i % 11, precision)
            for i in range(40)
        ],
        "mixed_spacing": [cast(rng.uniform(-1000, 1000)) for _ in range(40)],
    }


def weight_profiles():
    return {
        "integer": [float(1 + i % 9) for i in range(40)],
        "binary_fractional": [float(1 + i % 9) / 16 for i in range(40)],
        "decimal_fractional": [float(1 + i % 9) * 0.1 for i in range(40)],
        "large_integer": [float(2**24 + 1) if i % 3 == 0 else 1.0 for i in range(40)],
        "small_normal": [float(1 + i % 9) * 1e-100 for i in range(40)],
        "large_normal": [float(1 + i % 9) * 1e100 for i in range(40)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=202610140712)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-size", type=int, nargs="+")
    parser.add_argument("--surface", choices=["public", "native"], default="public")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_size is None:
        args.max_size = (
            [10, 20, 40, 100, 1000] if args.surface == "public" else [1, 2, 3]
        )
    if args.trials < 1 or any(size < 1 for size in args.max_size):
        parser.error("trials and max-size must be positive")
    if args.surface == "public" and any(
        size < 10 or size > 20000 for size in args.max_size
    ):
        parser.error(
            "public Python supports max-size 10..20000; small sizes are checked as expected validation cases"
        )
    digest_type = td.TDigest if args.surface == "public" else td._NativeTDigest
    started = time.monotonic()
    checks = Counter()
    failures = []
    complete = True

    def failure(kind, context, **details):
        checks["failures_" + kind] += 1
        if len(failures) < 200:
            failures.append({"kind": kind, **context, **details})

    if args.surface == "public":
        for precision in ["f32", "f64"]:
            for size in [1, 2, 3]:
                try:
                    td.TDigest.from_means_weights(
                        [0.0], [1.0], precision=precision, max_size=size
                    )
                except ValueError as error:
                    if str(error) == "max_size must be >= 10.":
                        checks["expected_public_capacity_rejections"] += 1
                    else:
                        failure(
                            "validation_error",
                            {"precision": precision, "max_size": size},
                            message=str(error),
                        )
                else:
                    failure(
                        "missing_capacity_validation",
                        {"precision": precision, "max_size": size},
                    )

    def verify(digest, values, weights, context):
        means = digest.means.tolist()
        mass = digest.weights.tolist()
        raw = digest.to_bytes()
        assert raw[:4] == b"TDIG" and raw[4] == 3, "Audit expects default TDIG v3"
        count, minimum, maximum = struct.unpack_from("<ddd", raw, 20)
        context = {
            **context,
            "values": values,
            "weights": weights,
            "centroid_means": means,
            "centroid_weights": mass,
        }
        checks["states"] += 1
        if not means or any(not math.isfinite(x) for x in means):
            failure("nonfinite_means", context)
            return
        if any(a >= b for a, b in zip(means, means[1:])):
            failure("unordered_means", context)
        if any(not math.isfinite(w) or w <= 0 for w in mass):
            failure("invalid_mass", context)
        if any(x < min(values) or x > max(values) for x in means):
            failure("centroid_outside_support", context)
        if (minimum, maximum) != (min(values), max(values)):
            failure(
                "extrema",
                context,
                actual=[minimum, maximum],
                expected=[min(values), max(values)],
            )
        exact = sum(map(Fraction, weights), Fraction())
        expected = float(exact)
        stored = math.fsum(mass)
        # Integer sums <= 2^53 and these small binary fractions have exact
        # positive additions, so they need no epsilon. Decimal/scaled inputs
        # use a 64-epsilon relative bound for <=40 inputs and merge stages.
        exact_mass = context["profile"] in {
            "integer",
            "binary_fractional",
            "large_integer",
        }
        for name, actual in [("count", count), ("centroid_total", stored)]:
            checks["mass_checks"] += 1
            if not math.isfinite(actual) or (
                actual != expected
                if exact_mass
                else not math.isclose(
                    actual, expected, rel_tol=64 * sys.float_info.epsilon
                )
            ):
                failure(
                    "mass",
                    context,
                    field=name,
                    actual=actual,
                    expected=expected,
                    exact=exact_mass,
                )
        support = [minimum, maximum]
        probes = set(values + means + support)
        probes.update(a * 0.5 + b * 0.5 for a, b in zip(means, means[1:]))
        probes.update(minimum * (1 - i / 32) + maximum * (i / 32) for i in range(33))
        probes.update(
            [math.nextafter(minimum, -math.inf), math.nextafter(maximum, math.inf)]
        )
        probes = sorted(probes)
        cdf = list(digest.cdf(probes))
        checks["cdf_values"] += len(cdf)
        for x, value in zip(probes, cdf):
            if not math.isfinite(value) or not 0 <= value <= 1:
                failure("cdf_bounds", context, probe=x, actual=value)
                break
        for i, (a, b) in enumerate(zip(cdf, cdf[1:])):
            if a > b:
                failure(
                    "cdf_monotonicity", context, probes=probes[i : i + 2], actual=[a, b]
                )
                break
        qs = (
            [0.0, math.nextafter(0.0, 1.0)]
            + [i / 128 for i in range(1, 128)]
            + [math.nextafter(1.0, 0.0), 1.0]
        )
        quantiles = list(digest.quantile(qs))
        checks["quantile_values"] += len(quantiles)
        for q, value in zip(qs, quantiles):
            if not math.isfinite(value) or not minimum <= value <= maximum:
                failure("quantile_support", context, q=q, actual=value)
                break
        for i, (a, b) in enumerate(zip(quantiles, quantiles[1:])):
            if a > b:
                failure(
                    "quantile_monotonicity", context, qs=qs[i : i + 2], actual=[a, b]
                )
                break
        if quantiles[0] != minimum or quantiles[-1] != maximum:
            failure("quantile_endpoints", context, actual=[quantiles[0], quantiles[-1]])
        if digest.to_bytes() != raw:
            failure("query_mutation", context)

    routes = [
        "batch",
        "reverse",
        "incremental_add",
        "merge40",
        "merge40_shuffled",
        "incremental_merge40",
    ]
    try:
        for trial in range(args.trials):
            for precision in ["f32", "f64"]:
                for label, values in datasets(args.seed, precision, trial).items():
                    for profile, weights in weight_profiles().items():
                        for size in args.max_size:
                            kwargs = {"max_size": size}
                            if args.surface == "public":
                                kwargs["precision"] = precision
                            else:
                                kwargs["f32_mode"] = precision == "f32"
                            for route in routes:
                                if time.monotonic() - started > args.max_seconds:
                                    complete = False
                                    raise TimeoutError(
                                        "Audit time budget reached; remaining cases not run"
                                    )
                                context = dict(
                                    trial=trial,
                                    precision=precision,
                                    dataset=label,
                                    profile=profile,
                                    max_size=size,
                                    route=route,
                                )
                                checks["cases_attempted"] += 1
                                try:
                                    if route in {"batch", "reverse"}:
                                        xs, ws = (
                                            (values, weights)
                                            if route == "batch"
                                            else (values[::-1], weights[::-1])
                                        )
                                        digest = digest_type.from_means_weights(
                                            xs, ws, **kwargs
                                        )
                                    elif route == "incremental_add":
                                        digest = digest_type.from_array([], **kwargs)
                                        for i, (x, w) in enumerate(
                                            zip(values, weights), 1
                                        ):
                                            digest.add_weighted(x, w)
                                            if i in [1, 2, 3, 10, 20]:
                                                verify(
                                                    digest,
                                                    values[:i],
                                                    weights[:i],
                                                    {**context, "step": i},
                                                )
                                    else:
                                        shards = [
                                            digest_type.from_means_weights(
                                                [x], [w], **kwargs
                                            )
                                            for x, w in zip(values, weights)
                                        ]
                                        if route == "merge40_shuffled":
                                            random.Random(args.seed + trial).shuffle(
                                                shards
                                            )
                                        if route == "incremental_merge40":
                                            digest = digest_type.from_array(
                                                [], **kwargs
                                            )
                                            for i, shard in enumerate(shards, 1):
                                                digest.merge(shard)
                                                if i in [1, 2, 3, 10, 20]:
                                                    verify(
                                                        digest,
                                                        values[:i],
                                                        weights[:i],
                                                        {**context, "step": i},
                                                    )
                                        else:
                                            digest = digest_type.merge_all(shards)
                                    verify(digest, values, weights, context)
                                    checks["cases_completed"] += 1
                                except BaseException as error:
                                    if isinstance(
                                        error, (KeyboardInterrupt, SystemExit)
                                    ):
                                        raise
                                    failure(
                                        "exception",
                                        {
                                            **context,
                                            "values": values,
                                            "weights": weights,
                                        },
                                        exception_type=type(error).__name__,
                                        message=str(error),
                                    )
    except TimeoutError:
        pass

    package_root = Path(td.__file__).parent
    provenance = {
        "script_sha256": sha256(__file__),
        "python": sys.version,
        "package": td.__file__,
        "native_sha256": {
            str(path): sha256(path) for path in package_root.rglob("*.so")
        },
    }
    if args.source_dir:
        provenance["git_head"] = subprocess.check_output(
            ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        provenance["source_sha256"] = {
            path: sha256(args.source_dir / path)
            for path in [
                "src/tdigest/centroids.rs",
                "src/tdigest/compressor.rs",
                "src/tdigest/tdigest.rs",
                "src/tdigest/cdf.rs",
                "src/tdigest/quantile.rs",
            ]
        }
        provenance["source_ref"] = subprocess.check_output(
            ["git", "-C", str(args.source_dir), "rev-parse", args.source_ref], text=True
        ).strip()
        provenance["source_ref_sha256"] = {
            path: hashlib.sha256(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(args.source_dir),
                        "show",
                        f"{args.source_ref}:{path}",
                    ]
                )
            ).hexdigest()
            for path in provenance["source_sha256"]
        }
        provenance["worktree_core_matches_source_ref"] = (
            provenance["source_sha256"] == provenance["source_ref_sha256"]
        )
    failed = any(
        name.startswith("failures_") and value for name, value in checks.items()
    )
    status = "INCOMPLETE" if not complete else "FAIL" if failed else "PASS"
    report = dict(
        status=status,
        seed=args.seed,
        trials=args.trials,
        max_sizes=args.max_size,
        surface=args.surface,
        seconds=time.monotonic() - started,
        checks=dict(checks),
        failures=failures,
        provenance=provenance,
        scope="Compressed-state invariants only. No exact-rank assertion after lossy compression. "
        "Weights have finite normal aggregate mass; subnormal-total and aggregate-overflow cases are separate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ["status", "seed", "seconds", "checks"]}
        )
    )
    return 2 if not complete else int(failed)


if __name__ == "__main__":
    sys.exit(main())
