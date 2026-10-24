"""Check the original API and published pickle migration in a release Linux wheel."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from check_upstream_sources import verify_sources


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verify_sources(root)
    project = root / "bindings/python"
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
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run(*args: str, cwd: Path = root) -> None:
        subprocess.run(args, cwd=cwd, env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="tdigest-wheel-test-") as directory:
        temporary = Path(directory)
        wheels = temporary / "wheels"
        run(
            "uv",
            "run",
            "--no-sync",
            "maturin",
            "build",
            "--release",
            "-F",
            "python",
            "--compatibility",
            "linux",
            "-o",
            str(wheels),
            cwd=project,
        )
        (wheel,) = wheels.glob("*.whl")
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
            shutil.copyfile(
                root / "bindings/python/tests/data" / name, parity / "data" / name
            )
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
