"""Record the actual host and artifact identities for an owner-run qualification."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import zipfile


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def physical_core_count(cores):
    """Unknown or negative topology identifiers cannot establish a core count."""
    try:
        identities = [(int(row["package"]), int(row["core"])) for row in cores]
    except (TypeError, ValueError):
        return None
    if not identities or any(package < 0 or core < 0 for package, core in identities):
        return None
    return len(set(identities))


def quota_paths(hierarchy):
    """Only inspect a mounted, contained cgroup-v2 hierarchy."""
    unified = next((line.split(":", 2)[2] for line in hierarchy.splitlines() if line.startswith("0::")), None)
    root = Path("/sys/fs/cgroup")
    if unified is None or not unified.startswith("/") or ".." in Path(unified).parts:
        return []
    current = root / unified.lstrip("/")
    paths = []
    while True:
        paths.append(current / "cpu.max")
        if current == root:
            return paths
        current = current.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--require-physical-cores", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new receipt path")
    if args.require_physical_cores < 0:
        parser.error("minimum core count cannot be negative")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    cores = []
    for cpu in allowed:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        cores.append(
            {
                "cpu": cpu,
                "package": read(root / "topology/physical_package_id"),
                "core": read(root / "topology/core_id"),
                "l3_id": read(root / "cache/index3/id"),
                "l3_size": read(root / "cache/index3/size"),
            }
        )
    physical = physical_core_count(cores)
    # Preserve raw quota/cgroup evidence. Missing mounted hierarchy means unknown.
    hierarchy = read("/proc/self/cgroup")
    quota = {"root_cpu_max": read("/sys/fs/cgroup/cpu.max"), "process_cgroup": hierarchy, "ancestors": []}
    if hierarchy:
        for path in quota_paths(hierarchy):
            value = read(path)
            if value is not None:
                quota["ancestors"].append({"path": str(path.parent), "cpu_max": value})
    quota["status"] = (
        "RECORDED_VISIBLE_FILES; hidden limits may exist"
        if quota["ancestors"] or quota["root_cpu_max"]
        else "UNAVAILABLE; do not assume unlimited CPU quota"
    )
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "visible_logical_cpus": len(allowed),
        "available_physical_cores": physical,
        "topology": cores,
        "cpu_quota": quota,
        "meminfo": read("/proc/meminfo"),
        "numa": [
            {"node": p.name, "cpulist": read(p / "cpulist")}
            for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
        ],
        "thread_environment": {
            name: os.environ.get(name)
            for name in [
                "RAYON_NUM_THREADS",
                "POLARS_MAX_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]
        },
        "modules": {},
    }
    for name in ["numpy", "polars", "pyarrow", "gr_tdigest", "tdigest_rs"]:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {
                "version": getattr(module, "__version__", None),
                "path": getattr(module, "__file__", None),
            }
        except ImportError as error:
            result["modules"][name] = {"unavailable": type(error).__name__}
    try:
        native = Path(importlib.import_module("gr_tdigest._gr_tdigest").__file__)
        result["native_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
        result["distribution"] = {"name": "gr-tdigest", "version": importlib.metadata.version("gr-tdigest")}
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pass
    if args.jar:
        result["jar"] = {"path": str(args.jar.resolve()), "sha256": hashlib.sha256(args.jar.read_bytes()).hexdigest()}
        with zipfile.ZipFile(args.jar) as archive:
            result["jar"]["manifest"] = archive.read("META-INF/MANIFEST.MF").decode()
    result["requested_minimum_physical_cores"] = args.require_physical_cores
    insufficient = args.require_physical_cores and (physical is None or physical < args.require_physical_cores)
    result["core_check"] = "UNTESTED_INSUFFICIENT_CORES" if insufficient else "PASS_RECORDED_HOST_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as receipt:
        receipt.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ["available_physical_cores", "visible_logical_cpus", "core_check"]}))
    if insufficient:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
