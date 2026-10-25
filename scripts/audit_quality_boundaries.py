"""Run exact boundary controls and retain known numerical failures explicitly."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def hashes(root):
    files = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "examples/quality_boundaries.rs",
        *sorted((root / "src").rglob("*.rs")),
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="check unchanged known outcomes; accuracy failures remain FAIL")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    before = hashes(root)
    # These exact numerical controls need no optimization. Reuse the dev
    # dependencies built by make build/test instead of compiling a release tree.
    command = ["cargo", "run", "--locked", "--profile", "dev", "--example", "quality_boundaries"]
    print("Boundary audit: reusing the dev build; compiler progress follows", file=sys.stderr, flush=True)
    start = time.perf_counter()
    run = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, check=True)
    elapsed = time.perf_counter() - start
    if hashes(root) != before:
        raise RuntimeError("Source changed while compiling boundary controls")
    report = json.loads(run.stdout)
    target = root / Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    binary = target / "debug/examples" / ("quality_boundaries.exe" if os.name == "nt" else "quality_boundaries")
    report.update(
        source_sha256=before,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        command=command,
        build_profile="dev",
        elapsed_seconds=elapsed,
    )
    same = False
    if args.baseline:
        same = report["controls"] == json.loads(args.baseline.read_text())["controls"]
    report["regression_status"] = "PASS" if same else "NOT_CHECKED" if not args.baseline else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("accuracy_status", "regression_status", "failed")}))
    print(f"Boundary audit completed in {elapsed:.2f}s", file=sys.stderr, flush=True)
    return 0 if same or report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
