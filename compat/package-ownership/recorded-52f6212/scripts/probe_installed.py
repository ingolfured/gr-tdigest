"""Inspect one stable installation in a fresh process; never build code."""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import pickle
import re
import struct
import sys


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def restored_states(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    states = {}
    for case in fixture["cases"]:
        payload = base64.b64decode(case["pickle_base64"], validate=True)
        require(hashlib.sha256(payload).hexdigest() == case["pickle_sha256"], "fixture pickle hash")
        digest = pickle.loads(payload)
        bits = lambda value: struct.pack(">d", float(value)).hex()
        actual = {
            "class_module": type(digest).__module__,
            "class_name": type(digest).__qualname__,
            "means_dtype": digest.means.dtype.str,
            "means_shape": list(digest.means.shape),
            "means_hex": digest.means.tobytes().hex(),
            "weights_dtype": digest.weights.dtype.str,
            "weights_hex": digest.weights.tobytes().hex(),
            "len": len(digest),
            "median_f64_bits": bits(digest.median()),
            "quantiles_f64_bits": [bits(digest.quantile(q)) for q in fixture["quantile_probes"]],
            "trimmed_mean_f64_bits": [bits(digest.trimmed_mean(a, b)) for a, b in fixture["trim_ranges"]],
        }
        require(actual == case["expected_old_restored"], f"golden mismatch: {case['id']}")
        states[case["id"]] = actual
    require(len(states) == 32, "expected 32 genuine published-pickle cases")
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["old", "upgraded", "reinstalled", "uninstalled"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--previous-record", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-native-sha256")
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    require(sys.flags.isolated == 1, "probe must use python -I")
    report = {"phase": args.phase, "python": sys.version, "prefix": str(prefix), "isolated": 1}
    if args.phase == "uninstalled":
        for package in ["tdigest_rs", "gr_tdigest"]:
            require(importlib.util.find_spec(package) is None, f"package remains importable: {package}")
        for name in ["tdigest-rs", "gr-tdigest"]:
            try:
                metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
            raise RuntimeError(f"distribution remains installed: {name}")
        previous = json.loads(args.previous_record.read_text())
        leftovers = [p for p in previous["owned_paths"] if Path(p).exists()]
        require(not leftovers, f"owned files remain after uninstall: {leftovers}")
        report.update({"packages_unavailable": True, "distributions_absent": True,
                       "recorded_owned_files_removed": len(previous["owned_paths"])})
    else:
        dist = metadata.distribution("tdigest-rs")
        require(normalized(dist.metadata["Name"]) == "tdigest-rs", "wrong distribution identity")
        require(dist.version == args.expected_version, f"unexpected version {dist.version}")
        try:
            metadata.distribution("gr-tdigest")
        except metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError("independent gr-tdigest distribution must be absent")
        owners = []
        for item in metadata.distributions():
            if any(str(p).startswith("tdigest_rs/") for p in item.files or []):
                owners.append({"name": item.metadata["Name"], "version": item.version})
        require(len(owners) == 1 and normalized(owners[0]["name"]) == "tdigest-rs", f"namespace owners: {owners}")
        package = importlib.import_module("tdigest_rs")
        shim = importlib.import_module("tdigest_rs.tdigest_rs")
        native = shim if args.phase == "old" else importlib.import_module("gr_tdigest._gr_tdigest")
        paths = {"package": Path(package.__file__).resolve(), "shim_or_original_native": Path(shim.__file__).resolve(),
                 "selected_native": Path(native.__file__).resolve()}
        require(all(p.is_relative_to(prefix) for p in paths.values()), f"outside-environment imports: {paths}")
        if args.phase == "old":
            require(paths["selected_native"].name == "tdigest_rs.abi3.so", "old ABI3 extension not selected")
        else:
            require(paths["shim_or_original_native"].name == "tdigest_rs.py", "new shim not selected")
            old_native = paths["shim_or_original_native"].with_name("tdigest_rs.abi3.so")
            require(not old_native.exists(), "stale original native extension shadows shim")
            report["old_native_removed"] = True
        native_sha = file_hash(paths["selected_native"])
        require(native_sha == args.expected_native_sha256, "selected native bytes differ from wheel")
        files = []
        owned_paths = []
        for entry in dist.files or []:
            actual = Path(dist.locate_file(entry)).resolve()
            require(actual.is_relative_to(prefix), f"RECORD path escapes disposable prefix: {entry}")
            files.append({"path": str(entry), "size": entry.size,
                          "record_hash": None if entry.hash is None else f"{entry.hash.mode}={entry.hash.value}"})
            owned_paths.append(str(actual))
        report.update({"distribution": {"name": dist.metadata["Name"], "version": dist.version,
                                         "requires_python": dist.metadata["Requires-Python"],
                                         "requires_dist": dist.metadata.get_all("Requires-Dist")},
                       "namespace_owners": owners, "paths": {k: str(v) for k, v in paths.items()},
                       "native_sha256": native_sha, "record": dist.read_text("RECORD"),
                       "files": files, "owned_paths": owned_paths,
                       "dependencies": {name: metadata.version(name) for name in ["numpy", "pip"]}})
        report["fixture_sha256"] = file_hash(args.fixture)
        report["restored_states"] = restored_states(args.fixture)
        if args.baseline:
            previous = json.loads(args.baseline.read_text())
            require(report["fixture_sha256"] == previous["fixture_sha256"], "fixture changed between phases")
            require(report["restored_states"] == previous["restored_states"], "actual old-restored state changed")
            report["exact_old_restored_states_matched"] = 32
        if args.phase != "old":
            import gr_tdigest as modern
            import numpy as np
            import polars as pl

            digest = modern.TDigest.from_means_weights(np.array([0.0, 10.0]), np.array([90.0, 10.0]))
            require(digest.median() == 0.0, "modern weighted-median smoke")
            require(modern.TDigest.from_bytes(digest.to_bytes()).to_bytes() == digest.to_bytes(), "modern wire smoke")
            frame = pl.DataFrame({"blob": [digest.to_bytes()]}).select(d=modern.from_bytes("blob", precision="f64"))
            require(frame.select(modern.median("d")).item() == 0.0, "Polars plugin smoke")
            report["modern_smoke"] = "weighted median, wire roundtrip and Polars plugin passed"
            report["dependencies"]["polars"] = metadata.version("polars")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "result": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
