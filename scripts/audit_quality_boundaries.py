"""Run exact boundary controls and retain known numerical failures explicitly."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def hashes(root):
    files = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "examples/quality_boundaries.rs",
        *sorted((root / "src").rglob("*.rs")),
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def compare_controls(controls, baseline):
    """Retain exact targets; allow a known failure to become an actual pass."""
    errors = []
    frozen = {row["id"]: row for row in baseline}
    actual = {row["id"]: row for row in controls}
    if len(frozen) != len(baseline) or len(actual) != len(controls):
        errors.append("duplicate control IDs")
    if set(frozen) != set(actual):
        errors.append("missing or unexpected control IDs")
    for ident in sorted(set(frozen) & set(actual)):
        old, new = frozen[ident], actual[ident]
        if new["expected"] != old["expected"]:
            errors.append(f"{ident}: expected target changed")
        elif new.get("status") == "PASS":
            observed = new.get("observed")
            if not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed != new["expected"]:
                errors.append(f"{ident}: PASS does not match exact target")
        elif new != old:
            errors.append(f"{ident}: new or changed numerical failure")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="check exact frozen targets; genuine fixes may pass, known failures remain explicit",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    before = hashes(root)
    command = ["cargo", "run", "--locked", "--quiet", "--release", "--example", "quality_boundaries"]
    run = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if hashes(root) != before:
        raise RuntimeError("Source changed while compiling boundary controls")
    report = json.loads(run.stdout)
    target = root / Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    binary = target / "release/examples" / ("quality_boundaries.exe" if os.name == "nt" else "quality_boundaries")
    report.update(source_sha256=before, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), command=command)
    errors = (
        compare_controls(report["controls"], json.loads(args.baseline.read_text())["controls"]) if args.baseline else []
    )
    report["regression_errors"] = errors
    report["regression_status"] = "NOT_CHECKED" if not args.baseline else "FAIL" if errors else "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("accuracy_status", "regression_status", "failed")}))
    return 0 if (not errors if args.baseline else report["failed"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
