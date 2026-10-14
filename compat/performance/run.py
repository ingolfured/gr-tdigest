#!/usr/bin/env python3
"""Build and run a Linux Rust-core comparison against verified original sources."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

PIN = "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="Independent G-Research/tdigest-rs git checkout containing the pinned commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to target/compat-performance/run-<UTC timestamp>",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only the 1,000-sample uniform case, f32 and f64",
    )
    parser.add_argument(
        "--cpu", type=int, help="Linux logical CPU; defaults to lowest allowed CPU"
    )
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error(
            "Run this benchmark on Linux/WSL; Windows measurements are outside the audit scope."
        )
    source_dir = Path(__file__).resolve().parent
    repo = source_dir.parent.parent
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output or repo / "target/compat-performance" / f"run-{stamp}"
    ).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream.resolve()
    if upstream == repo:
        parser.error(
            "Pass an independent G-Research/tdigest-rs checkout, not this repository."
        )

    def git_bytes(spec: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "show", f"{PIN}:{spec}"]
        )

    provenance = json.loads(
        (source_dir / "original-source-provenance.json").read_text()
    )
    original = output / "upstream-core"
    (original / "src").mkdir(parents=True)
    for name, expected in provenance["normalized_lf_sha256"].items():
        data = git_bytes(f"src/{name}").replace(b"\r\n", b"\n")
        if digest(data) != expected:
            raise RuntimeError(f"Pinned original source checksum mismatch: {name}")
        (original / "src" / name).write_bytes(data)
    manifest = tomllib.loads(git_bytes("Cargo.toml").decode())
    lines = [
        "[package]",
        'name="tdigest_core"',
        f"version={json.dumps(manifest['package']['version'])}",
        f"edition={json.dumps(manifest['package']['edition'])}",
        "[dependencies]",
    ]
    for name, version in manifest["dependencies"].items():
        if not isinstance(version, str):
            raise RuntimeError("Pinned upstream manifest dependency format changed")
        lines.append(f"{name}={json.dumps(version)}")
    (original / "Cargo.toml").write_text("\n".join(lines) + "\n")
    (output / "src").mkdir()
    shutil.copyfile(source_dir / "harness.rs", output / "src/main.rs")
    (output / "Cargo.toml").write_text(
        '[package]\nname="overnight-digest-bench"\nversion="0.0.0"\nedition="2021"\n'
        "[workspace]\n[dependencies]\n"
        f"gr-tdigest={{path={json.dumps(str(repo))}}}\n"
        'tdigest_core={path="upstream-core"}\nserde_json="1.0.145"\n'
    )
    shutil.copyfile(repo / "Cargo.lock", output / "Cargo.lock")
    env = os.environ.copy()
    target = Path(
        env.get("CARGO_TARGET_DIR", str(repo / "target/compat-performance/cargo"))
    ).resolve()
    env["CARGO_TARGET_DIR"] = str(target)
    env["RAYON_NUM_THREADS"] = "1"

    def command_output(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=repo, env=env, text=True).strip()

    hashes = {
        str(p.relative_to(repo)): digest(p.read_bytes())
        for p in (repo / "src/tdigest").rglob("*.rs")
    }
    metadata = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kernel": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        "rustc": command_output(["rustc", "-Vv"]),
        "repository_head": command_output(["git", "rev-parse", "HEAD"]),
        "repository_status": command_output(["git", "status", "--short"]),
        "source_sha256": hashes,
        "harness_sha256": digest((source_dir / "harness.rs").read_bytes()),
        "original_pin": PIN,
        "profile": "cargo release defaults: opt-level=3, no LTO, generic target unless overridden by environment",
        "scope": "Rust cores; original and modern share process jemalloc; excludes language adapters",
        "environment_overrides": {
            k: env.get(k)
            for k in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RUSTC",
                "RAYON_NUM_THREADS",
            )
        },
        "smoke": args.smoke,
    }
    with (output / "build.log").open("w") as log:
        subprocess.run(
            ["cargo", "build", "--release", "--offline", "--jobs", str(args.jobs)],
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if any(
        digest((repo / name).read_bytes()) != expected
        for name, expected in hashes.items()
    ):
        raise RuntimeError(
            "Digest source changed during compilation; rerun against a stable checkout"
        )
    affinity = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity})
    metadata["affinity_cpu"] = affinity
    binary = target / "release/overnight-digest-bench"
    metadata["binary_sha256"] = digest(binary.read_bytes())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (
        (output / "results.jsonl").open("w") as stdout,
        (output / "progress.log").open("w") as stderr,
    ):
        result = subprocess.run(
            [str(binary)] + (["--smoke"] if args.smoke else []),
            cwd=output,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    metadata["exit_status"] = result.returncode
    metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata["source_changed_during_run"] = [
        name
        for name, expected in hashes.items()
        if digest((repo / name).read_bytes()) != expected
    ]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {output}; exit={result.returncode}")
    if metadata["source_changed_during_run"]:
        raise RuntimeError(
            "Digest source changed during timing; see metadata before interpreting results"
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
