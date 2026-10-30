"""Fresh original-delta300 vs modern-capacity1000 comparison, not equal storage."""

from pathlib import Path
import argparse
import importlib.util
import importlib
import importlib.metadata
import json
import sys
import inspect
import hashlib
import subprocess
import numpy as np
from tdigest_rs.lib import DEFAULT_DELTA

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-sha", required=True, help="Library source identity from the artifact manifest")
parser.add_argument("--expected-native-sha256", help="Optional exact installed-artifact constraint")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = args.output
if output.exists():
    raise SystemExit("Refusing to overwrite a prior measurement")
spec = importlib.util.spec_from_file_location("_q", root / "scripts/quality_budgets.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
q = b.q

assert DEFAULT_DELTA == 300.0
assert inspect.signature(q.Original.from_array).parameters["delta"].default == 300.0
native = Path(importlib.import_module("gr_tdigest._gr_tdigest").__file__)
if args.expected_native_sha256 and q.sha(native) != args.expected_native_sha256:
    raise RuntimeError("Installed native hash differs from requested artifact")
original_wrapper = Path(importlib.import_module("tdigest_rs.lib").__file__)
assert q.sha(original_wrapper) == "0ed15e8d3d18e2d9e82b4a0c8e1cdbace3de315dc79649566e35c28bb9446aad"
record = {
    "kind": "Fresh explicit default-compression comparison; actual constructor/merge defaults; means precision controlled equally; not equal storage or performance test",
    "runtime_source_sha": args.source_sha,
    "check_checkout_base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (root / ".git").exists()
    else None,
    "check_working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    if (root / ".git").exists()
    else None,
    "native_sha256": q.sha(native),
    "original_source_sha": "8d9cf1fe82c5c9254e5dc9593f01717e317f4333",
    "package": importlib.metadata.version("gr-tdigest"),
    "original_wrapper_sha256": q.sha(original_wrapper),
    "runner_sha256": q.sha(Path(__file__)),
    "reference_runner_sha256": q.sha(Path(q.__file__)),
    "shape_runner_sha256": q.sha(Path(b.__file__)),
    "data_manifest_sha256": q.sha(root / "tests/quality/data/manifest.json"),
    "source_inputs_sha256": q.source_hashes(),
    "python": sys.version,
    "numpy": np.__version__,
    "configuration": {
        "original_delta": 300,
        "modern_max_size": 1000,
        "modern_scale": "k2",
        "modern_singletons": "use",
        "input_orders": ["natural", "reverse"],
        "paths": ["bulk", "left40", "balanced40"],
        "precisions": ["f32", "f64"],
        "comparison": "same cast observations/precision; default compression controls omitted; no matching or tuning by accuracy",
        "metrics": "Unchanged public exact reference; CDF original unavailable. Strict per-metric wins/losses plus existing one-observation/f32/f64 regression resolution; no new acceptance threshold",
    },
    "rows": [],
    "measurement_status": "RUNNING",
}


def build(values, precision, modern, route):
    def raw(a):
        return q.Modern.from_array(a, precision=precision) if modern else q.Original.from_array(a)

    def merge(a, b):
        return q.Modern.merge_all([a, b]) if modern else a.merge(b)

    if route == "bulk":
        return raw(values)
    parts = [raw(p) for p in np.array_split(values, min(40, len(values))) if len(p)]
    if route == "left40":
        out = parts[0]
        for part in parts[1:]:
            out = merge(out, part)
        return out
    while len(parts) > 1:
        parts = [merge(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


for cohort in ["capacity", "additional"]:
    for name, kind, raw in b.cases(root, cohort):
        for precision in ["f32", "f64"]:
            values = np.array(raw, dtype=np.float32 if precision == "f32" else np.float64)
            exact = q.Exact(values.tolist())
            for order in ["natural", "reverse"]:
                data = values if order == "natural" else values[::-1].copy()
                for route in ["bulk", "left40", "balanced40"]:
                    row = {
                        "id": f"{cohort}/{name}/{precision}/{order}/{route}",
                        "cohort": cohort,
                        "dataset": name,
                        "kind": kind,
                        "precision": precision,
                        "order": order,
                        "path": route,
                        "n": len(data),
                        "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                    }
                    try:
                        for engine, modern in [("modern", True), ("original", False)]:
                            d = build(data, precision, modern, route)
                            if modern:
                                assert d.max_size == 1000
                            row[engine] = q.score(d, exact, modern)
                            row[engine + "_state"] = {
                                "centroids": len(d),
                                "field_bytes": len(d) * (data.dtype.itemsize + (9 if modern else 4)),
                                **q.invariants(d, data, precision, modern),
                            }
                        row["strict_original_better"] = [
                            m for m in q.METRICS if row["modern"][m] > row["original"][m] + 1e-12
                        ]
                        row["original_better_beyond_existing_resolution"] = [
                            m
                            for m in q.METRICS
                            if row["modern"][m] > q.regression_limit(m, row["original"][m], len(data), precision)
                        ]
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                    record["rows"].append(row)
        q.atomic_json(output, record)
        print(cohort, name, "complete", flush=True)
q.verify_inputs_unchanged(record, native)
assert q.sha(original_wrapper) == record["original_wrapper_sha256"]
rows = [r for r in record["rows"] if "error" not in r]
record["summary"] = {
    "profiles": len(record["rows"]),
    "exceptions": len(record["rows"]) - len(rows),
    "metrics": {
        m: {
            "modern_better": sum(r["modern"][m] < r["original"][m] - 1e-12 for r in rows),
            "tied": sum(abs(r["modern"][m] - r["original"][m]) <= 1e-12 for r in rows),
            "original_better": sum(r["original"][m] < r["modern"][m] - 1e-12 for r in rows),
            "original_better_beyond_existing_resolution": sum(
                m in r["original_better_beyond_existing_resolution"] for r in rows
            ),
            "worst_modern_minus_original": max(
                (
                    {
                        "id": r["id"],
                        "modern": r["modern"][m],
                        "original": r["original"][m],
                        "difference": r["modern"][m] - r["original"][m],
                    }
                    for r in rows
                ),
                key=lambda x: x["difference"],
            ),
        }
        for m in q.METRICS
    },
    "modern_cdf_max": max(r["modern"]["cdf_max"] for r in rows),
    "profiles_with_original_better_beyond_existing_resolution": sum(
        bool(r["original_better_beyond_existing_resolution"]) for r in rows
    ),
}
record["measurement_status"] = "COMPLETED"
record["input_status"] = "UNCHANGED"
q.atomic_json(output, record)
print(json.dumps(record["summary"], indent=2))

if record["summary"]["exceptions"]:
    raise SystemExit(1)
