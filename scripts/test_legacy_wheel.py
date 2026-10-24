"""Check the original API and published pickle migration in a release Linux wheel."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from check_python_sdist import verify_sdist_contents
from check_upstream_sources import verify_sources


def verify_wheel_contents(wheel: Path) -> None:
    """Reject generated Python caches before installing the release artifact."""
    with ZipFile(wheel) as archive:
        unexpected = sorted(
            name
            for name in archive.namelist()
            if "__pycache__" in PurePosixPath(name).parts or PurePosixPath(name).suffix in {".pyc", ".pyo"}
        )
    if unexpected:
        raise RuntimeError("wheel contains Python cache/bytecode files: " + ", ".join(unexpected))
    print("Release wheel contains no Python cache/bytecode files.", flush=True)


def verify_upstream_snapshot(root: Path) -> None:
    verify_sources(root)
    upstream = root / "compat/tdigest-rs-upstream"
    checksums = re.findall(
        r"^([0-9a-f]{64})  (.+)$",
        (upstream / "PROVENANCE.md").read_text(),
        re.MULTILINE,
    )
    if len(checksums) != 7:
        raise RuntimeError("expected checksums for all seven unchanged upstream files")
    for expected, relative in checksums:
        if hashlib.sha256((upstream / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"copied upstream file changed: {relative}")
    print("All seven upstream file checksums match the pinned snapshot.", flush=True)


def stage_compatibility_tests(root: Path, temporary: Path) -> tuple[Path, Path]:
    tests = temporary / "tests"
    shutil.copytree(
        root / "compat/tdigest-rs-upstream/bindings/python/tests",
        tests,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    # Run exact outputs, API regressions, and genuine published pickles from the wheel.
    parity = temporary / "parity"
    (parity / "data").mkdir(parents=True)
    for name in (
        "test_original_numerics.py",
        "test_legacy_compatibility.py",
        "test_optional_polars_import.py",
        "test_published_pickle_migration.py",
    ):
        shutil.copyfile(root / "bindings/python/tests" / name, parity / name)
    for name in (
        "upstream_numerical_golden.json",
        "published_0_1_0_pickle_golden.json",
    ):
        shutil.copyfile(root / "bindings/python/tests/data" / name, parity / "data" / name)
    return tests, parity


def build_and_check_sdist(root: Path, output: Path, run: Callable[..., object]) -> Path:
    """Create the real source archive and check that Maturin preserves Cargo.lock."""
    lockfile = root / "Cargo.lock"
    before = lockfile.read_bytes()
    run(
        "uv",
        "run",
        "--project",
        str(root / "bindings/python"),
        "--no-sync",
        "maturin",
        "sdist",
        "-o",
        str(output),
        cwd=root / "bindings/python",
    )
    if lockfile.read_bytes() != before:
        raise RuntimeError("maturin sdist changed Cargo.lock; review the resolution before building")
    (sdist,) = output.glob("*.tar.gz")
    verify_sdist_contents(sdist, root)
    return sdist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="reuse this release wheel without rebuilding")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    project = root / "bindings/python"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run(*args: str, cwd: Path = root) -> None:
        subprocess.run(args, cwd=cwd, env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="tdigest-wheel-test-") as directory:
        temporary = Path(directory)
        if args.wheel:
            wheel = args.wheel.resolve(strict=True)
        else:
            build_and_check_sdist(root, temporary / "sdist", run)
            wheels = temporary / "wheels"
            run(
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "build",
                "--release",
                "--locked",
                "-F",
                "python",
                "--compatibility",
                "linux",
                "-o",
                str(wheels),
                cwd=project,
            )
            (wheel,) = wheels.glob("*.whl")
        verify_wheel_contents(wheel)
        requirements = temporary / "requirements.txt"
        run(
            "uv",
            "export",
            "--project",
            str(project),
            "--frozen",
            "--all-groups",
            "--no-emit-project",
            "--no-hashes",
            "--output-file",
            str(requirements),
        )
        venv = temporary / "venv"
        run("uv", "venv", "--python", sys.executable, str(venv))
        python = str(venv / "bin/python")
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "-r",
            str(requirements),
            str(wheel),
        )
        tests, parity = stage_compatibility_tests(root, temporary)
        # Source checkout must not satisfy these imports.
        run(
            python,
            "-I",
            "-c",
            "import pathlib, sys, tdigest_rs, gr_tdigest._gr_tdigest as native; "
            "prefix = pathlib.Path(sys.prefix).resolve(); "
            "assert pathlib.Path(tdigest_rs.__file__).resolve().is_relative_to(prefix); "
            "assert pathlib.Path(native.__file__).resolve().is_relative_to(prefix); "
            "print('Installed legacy package:', tdigest_rs.__file__); "
            "print('Installed Rust extension:', native.__file__)",
            cwd=temporary,
        )
        run(python, "-I", "-m", "pytest", "-q", str(tests), str(parity), cwd=temporary)


if __name__ == "__main__":
    main()
