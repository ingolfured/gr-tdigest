"""Attribute observed default-setting value/median differences without changing gates."""

from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comparison", type=Path, default=Path(__file__).with_name("defaults.json.gz"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output
if out.exists():
    raise SystemExit("Output exists")
spec = importlib.util.spec_from_file_location("_b", root / "scripts/quality_budgets.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
q = b.q
prior = q.read_json(args.comparison)
raw = {(c, n): v for c in ["capacity", "additional"] for n, k, v in b.cases(root, c)}
rows = []
for row in prior["rows"]:
    if not row["original_better_beyond_existing_resolution"]:
        continue
    data = np.array(raw[row["cohort"], row["dataset"]], dtype=np.float32 if row["precision"] == "f32" else np.float64)
    if row["order"] == "reverse":
        data = data[::-1].copy()
    exact = q.Exact(data.tolist())
    m = q.build(data, row["precision"], 1000, "modern", row["path"])
    o = q.build(data, row["precision"], 300, "original", row["path"])
    assert q.score(m, exact, True) == row["modern"] and q.score(o, exact, False) == row["original"]
    probes = []
    for probability in q.Q:
        mv = float(m.quantile(probability))
        ov = float(o.quantile(probability))
        ref = exact.quantile(probability)
        mr = exact.rank_error(probability, mv)
        orr = exact.rank_error(probability, ov)
        probes.append(
            {
                "q": probability,
                "modern": mv,
                "original": ov,
                "type7_reference": ref,
                "modern_abs_value_error": abs(mv - ref),
                "original_abs_value_error": abs(ov - ref),
                "modern_rank_error": mr,
                "original_rank_error": orr,
                "modern_returns_observed_point": exact.rank_interval(mv)[1] > exact.rank_interval(mv)[0],
            }
        )
    worst = max(probes, key=lambda p: p["modern_abs_value_error"])
    loss = [
        p
        for p in probes
        if p["modern_abs_value_error"]
        > p["original_abs_value_error"] + max(1e-12, 2e-6 if row["precision"] == "f32" else 1e-12) * exact.value_scale
    ]
    result = {
        "id": row["id"],
        "losing_metrics": row["original_better_beyond_existing_resolution"],
        "worst_modern_value_probe": worst,
        "value_losing_probes": len(loss),
        "value_losing_probes_with_zero_modern_rank_error": sum(p["modern_rank_error"] == 0 for p in loss),
        "value_losing_probes_with_modern_rank_no_worse": sum(
            p["modern_rank_error"] <= p["original_rank_error"] + 1e-12 for p in loss
        ),
        "value_losing_probes_with_strictly_worse_modern_rank": sum(
            p["modern_rank_error"] > p["original_rank_error"] + 1e-12 for p in loss
        ),
        "median_probe": next(p for p in probes if p["q"] == 0.5),
        "all_modern_value_losing_probes": loss,
    }
    rows.append(result)
receipt = {
    "comparison_sha256": q.sha(args.comparison),
    "runner_sha256": q.sha(Path(__file__)),
    "reference_runner_sha256": q.sha(Path(q.__file__)),
    "shape_runner_sha256": q.sha(Path(b.__file__)),
    "kind": "Fresh query-level reconstruction of the recorded default profiles with losses; attribution evidence, not a new acceptance gate",
    "profiles_reconstructed": len(rows),
    "value_losing_probes": sum(r["value_losing_probes"] for r in rows),
    "value_losing_probes_with_zero_modern_rank_error": sum(
        r["value_losing_probes_with_zero_modern_rank_error"] for r in rows
    ),
    "value_losing_probes_with_modern_rank_no_worse": sum(
        r["value_losing_probes_with_modern_rank_no_worse"] for r in rows
    ),
    "value_losing_probes_with_strictly_worse_modern_rank": sum(
        r["value_losing_probes_with_strictly_worse_modern_rank"] for r in rows
    ),
    "rows": rows,
}
q.atomic_json(out, receipt)
print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
print(
    "worst modern error point/rankexact:",
    sum(
        r["worst_modern_value_probe"]["modern_returns_observed_point"]
        and r["worst_modern_value_probe"]["modern_rank_error"] == 0
        for r in rows
    ),
)
for r in rows:
    if r["value_losing_probes_with_strictly_worse_modern_rank"] or "median_rank" in r["losing_metrics"]:
        print(r["id"], r["value_losing_probes_with_strictly_worse_modern_rank"], r["median_probe"])
