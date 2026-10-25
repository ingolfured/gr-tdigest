"""Reproducible accuracy checks against exact data and the preserved original engine.

Run with an installed gr-tdigest wheel. No data downloads occur during tests.
Logical field bytes are a comparison budget, not an estimate of heap/RSS use.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from statistics import NormalDist
import sys
import time

import numpy as np
from gr_tdigest import TDigest as Modern
from tdigest_rs import TDigest as Original

PROTOCOL = {
    "version": 1,
    "modern": {"scale": "k2", "singleton_policy": "use"},
    "budgets": {"fast": [64], "extended": [32, 128, 512]},
    "precisions": ["f32", "f64"],
    "paths": {"fast": ["bulk"], "extended": ["bulk", "left40", "balanced40"]},
    "orders": {"fast": ["natural"], "extended": ["natural", "reverse"]},
    "synthetic_rows": {"fast": 512, "extended": 8192},
    "real_rows": {"fast": "512 systematic rows over the complete timeline", "extended": "every source row"},
    "reference": "sorted observed input after precision casting; type7 quantile, ECDF rank interval, midpoint CDF",
    "budget": "modern centroid logical fields: mean4/8 + weight8 + kind1; original mean4/8 + weight4. No allocator, capacity or metadata claim.",
    "original_selection": "largest observed count within modern field-byte budget; count-only doubling/bisection, no accuracy selection",
    "comparison_eligible": "original occupies at least90% and at most100% of modern logical field bytes",
    "baseline_gate": "no relative deterioration allowance; rank metrics tolerate one observation, other metrics 2e-6(f32)/1e-12(f64); every metric separately",
    "probability_grid": "135 fixed probabilities: i/128 plus .0001,.001,.01,.99,.999,.9999",
}
Q = sorted(set([i / 128 for i in range(129)] + [0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999]))
METRICS = ("rank_mean", "rank_max", "median_rank", "tail_rank_max", "value_iqr_mean", "value_iqr_max")
GATED = (*METRICS, "cdf_max", "cdf_mean")


def regression_limit(metric: str, frozen: float, n: int, precision: str) -> float:
    tolerance = 1 / n + 1e-12 if metric in METRICS[:4] else 2e-6 if precision == "f32" else 1e-12
    return frozen + tolerance


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "rust-toolchain.toml",
        root / "bindings/python/pyproject.toml",
        *sorted((root / "src").rglob("*.rs")),
        *sorted((root / "bindings/python/gr_tdigest").glob("*.py")),
        *sorted((root / "bindings/python/tdigest_rs").glob("*.py")),
    ]
    return {str(p.relative_to(root)): sha(p) for p in paths}


@dataclass
class Exact:
    values: list[float]

    def __post_init__(self):
        self.values = sorted(map(float, self.values))
        if not self.values or not all(map(math.isfinite, self.values)):
            raise ValueError("Reference requires finite nonempty data")

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q outside [0,1]")
        position = q * (len(self.values) - 1)
        left = int(position)
        right = min(left + 1, len(self.values) - 1)
        fraction = position - left
        return self.values[left] * (1 - fraction) + self.values[right] * fraction

    def rank_interval(self, x: float) -> tuple[float, float]:
        n = len(self.values)
        return bisect_left(self.values, x) / n, bisect_right(self.values, x) / n

    def rank_error(self, q: float, x: float) -> float:
        lo, hi = self.rank_interval(x)
        return max(lo - q, q - hi, 0.0)

    def cdf(self, x: float) -> float:
        lo, hi = self.rank_interval(x)
        return (lo + hi) / 2

    @property
    def value_scale(self) -> float:
        iqr = self.quantile(0.75) - self.quantile(0.25)
        return iqr or (self.values[-1] - self.values[0]) or 1.0


def score(digest, exact: Exact, modern: bool) -> dict:
    predicted = [float(digest.quantile(q)) for q in Q]
    if not all(map(math.isfinite, predicted)):
        raise AssertionError("nonfinite quantile")
    tolerance = max(1.0, abs(exact.values[0]), abs(exact.values[-1])) * (
        2e-6 if digest.means.dtype == np.float32 else 2e-12
    )
    bounds = sum(x < exact.values[0] - tolerance or x > exact.values[-1] + tolerance for x in predicted)
    inversions = sum(a > b + tolerance for a, b in zip(predicted, predicted[1:]))
    if modern and (bounds or inversions):
        raise AssertionError("modern quantile bounds/monotonicity failure")
    ranks = [exact.rank_error(q, x) for q, x in zip(Q, predicted)]
    errors = [abs(x - exact.quantile(q)) for q, x in zip(Q, predicted)]
    tails = [(q, e) for q, e in zip(Q, ranks) if q <= 0.01 or q >= 0.99]
    result = dict(
        rank_mean=math.fsum(ranks) / len(ranks),
        rank_max=max(ranks),
        median_rank=ranks[Q.index(0.5)],
        tail_rank_max=max(e for _, e in tails),
        tail_relative_rank_max=max(e / max(min(q, 1 - q), 1 / len(exact.values)) for q, e in tails),
        value_abs_mean=math.fsum(errors) / len(errors),
        value_abs_max=max(errors),
        value_iqr_mean=math.fsum(errors) / len(errors) / exact.value_scale,
        value_iqr_max=max(errors) / exact.value_scale,
    )
    result.update(bounds_violations=bounds, monotonicity_violations=inversions)
    if modern:
        probes = sorted(set(exact.quantile(q) for q in Q))
        cdf = [float(digest.cdf(x)) for x in probes]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in cdf):
            raise AssertionError("invalid CDF")
        if any(a > b + 1e-12 for a, b in zip(cdf, cdf[1:])):
            raise AssertionError("nonmonotone CDF")
        cdf_errors = [abs(v - exact.cdf(x)) for x, v in zip(probes, cdf)]
        result.update(
            cdf_mean=math.fsum(cdf_errors) / len(cdf_errors), cdf_max=max(cdf_errors), cdf_probe_count=len(probes)
        )
    else:
        result.update(cdf_mean=None, cdf_max=None, cdf_probe_count=0)
    return result


def datasets(directory: Path, mode: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    for source in manifest:
        file = directory / source["derived_file"]
        if sha(file) != source["derived_sha256"]:
            raise RuntimeError("Pinned dataset checksum mismatch: " + file.name)
        with gzip.open(file, "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != source["rows"]:
            raise RuntimeError("Pinned dataset row count mismatch")
        for column in source["columns"]:
            values = [float(row[column]) for row in rows]
            if mode == "fast":
                values = [values[i * (len(values) - 1) // 511] for i in range(512)]
            yield source["id"] + "/" + column, "real", values
    n = PROTOCOL["synthetic_rows"][mode]
    u = [(i + 0.5) / n for i in range(n)]
    normal = [NormalDist().inv_cdf(p) for p in u]
    cases = {
        "constant": [7.0] * n,
        "uniform_grid": list(map(float, range(n))),
        "gaussian_stratified": normal,
        "lognormal_stratified": [math.exp(x) for x in normal],
        "exponential_stratified": [-math.log1p(-p) for p in u],
        "pareto_stratified": [(1 - p) ** (-2 / 3) for p in u],
        "bimodal": [x + (-5 if i % 2 else 5) for i, x in enumerate(normal)],
        "rounded_ties": [round(x, 1) for x in normal],
        "zero_inflated": [0.0 if i < n * 0.98 else float(i * i) for i in range(n)],
        "skewed_two_point": [0.0 if i < n * 0.9 else 1.0 for i in range(n)],
        "drift": [float((i // max(1, n // 40)) ** 2) + (i % 7) / 7 for i in range(n)],
        "signed_scale": [(-1 if i % 2 else 1) * math.exp(x * 4) for i, x in enumerate(normal)],
    }
    for name, values in cases.items():
        yield "synthetic/" + name, "synthetic", values


def build(values, precision: str, budget: float, engine: str, path: str):
    if engine == "modern":

        def raw(a):
            return Modern.from_array(a, max_size=int(budget), precision=precision, **PROTOCOL["modern"])

        def merge(a, b):
            return Modern.merge_all([a, b])
    else:

        def raw(a):
            return Original.from_array(a, delta=budget)

        def merge(a, b):
            return a.merge(b, delta=budget)

    if path == "bulk":
        return raw(values)
    parts = [raw(part) for part in np.array_split(values, min(40, len(values))) if len(part)]
    if path == "left40":
        result = parts[0]
        for part in parts[1:]:
            result = merge(result, part)
        return result
    while len(parts) > 1:
        parts = [merge(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


def match_original(values, precision: str, target_bytes: int, path: str):
    width = values.dtype.itemsize + 4
    trace = []
    candidates = []
    observed = []

    def attempt(delta):
        digest = build(values, precision, delta, "original", path)
        size = len(digest) * width
        trace.append({"delta": delta, "centroids": len(digest), "field_bytes": size})
        observed.append((size, delta, digest))
        if size <= target_bytes:
            candidates.append((size, delta, digest))
        return size

    lo, hi = 1.0, 2.0
    attempt(lo)
    while attempt(hi) <= target_bytes and hi < 65536:
        lo, hi = hi, hi * 2
        if len(candidates[-1][2]) == len(set(map(float, values))):
            break
    for _ in range(12):
        mid = (lo + hi) / 2
        if attempt(mid) <= target_bytes:
            lo = mid
        else:
            hi = mid
    # Some degenerate original states retain two or more identical means.
    # Keep them visible when even the smallest observed state exceeds budget.
    size, delta, digest = (
        max(candidates, key=lambda c: (c[0], c[1])) if candidates else min(observed, key=lambda c: c[0])
    )
    return digest, delta, size, trace


def invariants(digest, values, precision: str, modern: bool):
    means = list(map(float, digest.means))
    weights = list(map(float, digest.weights))
    if not means or len(means) != len(weights) or not all(math.isfinite(x) for x in means):
        raise AssertionError("invalid centroids")
    inversions = sum(a > b for a, b in zip(means, means[1:]))
    if modern and any(a >= b for a, b in zip(means, means[1:])):
        raise AssertionError("modern centroid means not strictly increasing")
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise AssertionError("invalid centroid weight")
    mass = math.fsum(weights)
    if abs(mass - len(values)) > len(values) * 1e-12:
        raise AssertionError(f"input mass lost: {mass} != {len(values)}")
    moment = math.fsum(m * w for m, w in zip(means, weights))
    expected = math.fsum(map(float, values))
    scale = max(1.0, math.fsum(abs(float(x)) for x in values))
    moment_relative_abs = abs(moment - expected) / scale
    moment_limit = 2e-6 if precision == "f32" else 2e-12
    if modern and moment_relative_abs > moment_limit:
        raise AssertionError(f"weighted contribution drift: {moment_relative_abs}")
    if modern:
        if digest.inner_kind() != precision:
            raise AssertionError("wrong actual storage precision")
        restored = Modern.from_bytes(digest.to_bytes())
        if restored.to_bytes() != digest.to_bytes():
            raise AssertionError("state roundtrip changed")
    return {
        "mass": mass,
        "moment_abs_error": abs(moment - expected),
        "moment_scaled_abs_error": moment_relative_abs,
        "moment_limit_exceeded": moment_relative_abs > moment_limit,
        "centroid_order_inversions": inversions,
    }


def atomic_json(path: Path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(rows):
    groups = {}
    for kind in ("all", "real", "synthetic"):
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        # Poor modern results remain in win/loss tables; removing accuracy-limit
        # failures would bias the comparison in favor of this implementation.
        eligible = [r for r in selected if r.get("comparison_eligible") and "modern" in r and "original" in r]
        comparison = {}
        for metric in METRICS:
            counts = dict(modern_better=0, tied=0, original_better=0)
            for row in eligible:
                a, b = row["modern"][metric], row["original"][metric]
                tolerance = 1e-12
                key = "tied" if abs(a - b) <= tolerance else "modern_better" if a < b else "original_better"
                counts[key] += 1
            comparison[metric] = counts
        groups[kind] = {
            "pairs": len(selected),
            "eligible_pairs": len(eligible),
            "errors": sum("error" in r for r in selected),
            "accuracy_limit_failures": sum(bool(r.get("accuracy_limits")) for r in selected),
            "original_moment_limit_failures": sum(
                r.get("original_state", {}).get("moment_limit_exceeded", False) for r in selected
            ),
            "metrics": comparison,
        }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast", "extended"), default="fast")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "tests/quality/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--known-failures", type=Path, help="explicit regression gate; retains FAIL accuracy status")
    args = parser.parse_args()
    if args.known_failures and not args.baseline:
        parser.error("--known-failures requires a frozen --baseline")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    native = Path(sys.modules["gr_tdigest._gr_tdigest"].__file__)
    receipt = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(),
        "runner_sha256": sha(Path(__file__)),
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "native_sha256": sha(native),
        "python": sys.version,
        "package": importlib.metadata.version("gr-tdigest"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_inputs_sha256": source_hashes(),
        "mode": args.mode,
        "rows": [],
    }
    baseline = {}
    if args.baseline:
        frozen = read_json(args.baseline)
        if (
            frozen["protocol_sha256"] != receipt["protocol_sha256"]
            or frozen["data_manifest_sha256"] != receipt["data_manifest_sha256"]
        ):
            raise RuntimeError("Baseline protocol/data changed; review the measurements before replacing it")
        baseline = {r["id"]: r for r in frozen["rows"]}
        if not baseline or len(baseline) != len(frozen["rows"]):
            raise RuntimeError("Baseline must contain unique, nonempty cases")
    for name, kind, raw in datasets(args.data, args.mode):
        for precision in PROTOCOL["precisions"]:
            values = np.array(raw, dtype=np.float32 if precision == "f32" else np.float64)
            exact = Exact(values.tolist())
            for budget in PROTOCOL["budgets"][args.mode]:
                for order in PROTOCOL["orders"][args.mode]:
                    data = values if order == "natural" else values[::-1].copy()
                    for path in PROTOCOL["paths"][args.mode]:
                        ident = f"{name}/{precision}/size{budget}/{order}/{path}"
                        row = {
                            "id": ident,
                            "dataset": name,
                            "kind": kind,
                            "precision": precision,
                            "max_size": budget,
                            "order": order,
                            "path": path,
                            "n": len(data),
                        }
                        try:
                            modern = build(data, precision, budget, "modern", path)
                            modern_bytes = len(modern) * (data.dtype.itemsize + 8 + 1)
                            row["modern_state"] = dict(
                                centroids=len(modern),
                                field_bytes=modern_bytes,
                                wire_bytes=len(modern.to_bytes()),
                                **invariants(modern, data, precision, True),
                            )
                            row["modern"] = score(modern, exact, True)
                            original, delta, size, trace = match_original(data, precision, modern_bytes, path)
                            row["original_state"] = dict(
                                centroids=len(original),
                                field_bytes=size,
                                delta=delta,
                                selection_trace=trace,
                                **invariants(original, data, precision, False),
                            )
                            row["original"] = score(original, exact, False)
                            row["comparison_eligible"] = 0.9 <= size / modern_bytes <= 1.0
                            row["budget_fill"] = size / modern_bytes
                            # Hard accuracy envelope is intentionally independent
                            # of which implementation wins this comparison.
                            bound = 0.1 if budget <= 32 else 0.05
                            row["accuracy_limits"] = [
                                metric for metric in ("rank_max", "cdf_max") if row["modern"][metric] > bound
                            ]
                            if baseline:
                                if ident not in baseline:
                                    raise AssertionError("case absent from frozen baseline")
                                for metric in GATED:
                                    limit = regression_limit(
                                        metric, baseline[ident]["modern"][metric], len(data), precision
                                    )
                                    if row["modern"][metric] > limit:
                                        raise AssertionError("quality regression: " + metric)
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                        receipt["rows"].append(row)
        print(name, "completed", flush=True)
        atomic_json(args.output / "results.json", receipt)
    if baseline and set(baseline) != {r["id"] for r in receipt["rows"]}:
        raise RuntimeError("Baseline/case coverage differs")
    receipt["elapsed_seconds"] = time.perf_counter() - start
    receipt["summary"] = summarize(receipt["rows"])
    receipt["facets"] = {
        field: {
            str(value): summarize([r for r in receipt["rows"] if r[field] == value])["all"]
            for value in sorted({r[field] for r in receipt["rows"]})
        }
        for field in ("precision", "max_size", "path", "dataset")
    }
    findings = {r["id"]: r["accuracy_limits"] for r in receipt["rows"] if r.get("accuracy_limits")}
    errors = any("error" in r for r in receipt["rows"])
    receipt["accuracy_status"] = "FAIL" if findings or errors else "PASS"
    known_matches = False
    if args.known_failures:
        known = json.loads(args.known_failures.read_text())
        known_matches = known["protocol_sha256"] == receipt["protocol_sha256"] and known["findings"] == findings
    receipt["regression_status"] = (
        "FAIL" if errors or (findings and not known_matches) or (args.known_failures and not known_matches) else "PASS"
    )
    receipt["status"] = (
        "KNOWN_LIMITS" if receipt["regression_status"] == "PASS" and findings else receipt["regression_status"]
    )
    atomic_json(args.output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "accuracy_status": receipt["accuracy_status"],
                "regression_status": receipt["regression_status"],
                "seconds": receipt["elapsed_seconds"],
                "summary": receipt["summary"],
            },
            indent=2,
        )
    )
    return 0 if receipt["regression_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
