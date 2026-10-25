"""Build one release wheel from the sdist and check CPython3.10-3.15 runtimes.

Linux gate. --wheel reuses a real release artifact without rebuilding it.
The matrix intentionally pins tested dependency combinations; the project lock
continues to govern the development environment, not these independent cells.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from check_python_sdist import verify_sdist_contents
from test_legacy_wheel import (
    build_and_check_sdist,
    stage_compatibility_tests,
    verify_upstream_snapshot,
    verify_wheel_contents,
)

MATRIX = (
    ("3.10", "1.26.4", "1.15.3", "8.4.2"),
    ("3.10", "2.2.6", "1.15.3", "8.4.2"),
    ("3.11", "1.26.4", "1.17.1", "8.4.2"),
    ("3.11", "2.3.4", "1.17.1", "8.4.2"),
    ("3.12", "1.26.4", "1.17.1", "8.4.2"),
    ("3.12", "2.3.4", "1.17.1", "8.4.2"),
    ("3.13", "2.3.4", "1.17.1", "8.4.2"),
    ("3.14", "2.5.4", "1.18.1", "9.1.1"),
    ("3.15", "2.5.4", "1.18.1", "9.1.1"),
)
MODERN_TESTS = (
    "bindings/python/tests/test_api_python.py",
    "bindings/python/tests/test_api_polars.py",
    "bindings/python/tests/test_day_numpy_input.py",
    "bindings/python/tests/test_day_numpy_output.py",
    "bindings/python/tests/test_digest_regressions.py",
    "integration/api_coherence/test_cluster_mean_bounds.py",
    "integration/api_coherence/test_subnormal_query_coherence.py",
    "integration/api_coherence/test_transactional_weight_scaling.py",
    "integration/api_coherence/test_k2norm_domain.py",
)

# Run by the installed cell interpreter with -I, outside the source checkout.
INSPECT = """
import hashlib, importlib, importlib.metadata as metadata, importlib.util, json, pathlib, sys
expected = json.loads(sys.argv[1])
assert sys.flags.isolated == 1
assert '.'.join(map(str, sys.version_info[:2])) == expected['python']
prefix = pathlib.Path(sys.prefix).resolve()
names = ('tdigest_rs', 'tdigest_rs.lib', 'tdigest_rs.tdigest_rs',
         'gr_tdigest', 'gr_tdigest._gr_tdigest')
modules = {name: importlib.import_module(name) for name in names}
paths = {name: pathlib.Path(module.__file__).resolve() for name, module in modules.items()}
assert all(path.is_relative_to(prefix) for path in paths.values()), paths
assert paths['tdigest_rs.tdigest_rs'].name == 'tdigest_rs.py', paths
assert modules['tdigest_rs'].TDigest.__module__ == 'tdigest_rs.lib'
native_hash = hashlib.sha256(paths['gr_tdigest._gr_tdigest'].read_bytes()).hexdigest()
assert native_hash == expected['native_sha256'], native_hash
dist = metadata.distribution(expected['distribution'])
assert dist.version == expected['version'], dist.version
assert sorted(part.strip() for part in dist.metadata['Requires-Python'].split(',')) == ['<3.16', '>=3.10'], dist.metadata['Requires-Python']
versions = {name: metadata.version(name) for name in ('numpy', 'scipy', 'pytest', 'pip')}
if expected['polars'] is None:
    assert importlib.util.find_spec('polars') is None
    assert not any(name == 'polars' or name.startswith('polars.') for name in sys.modules)
    try:
        metadata.distribution('polars')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError('base installation unexpectedly includes Polars')
else:
    versions['polars'] = metadata.version('polars')
    assert versions['polars'] == expected['polars']
for name in ('numpy', 'scipy', 'pytest'):
    assert versions[name] == expected[name], (name, versions[name])
print(json.dumps({'python': sys.version, 'executable': sys.executable,
                  'dependencies': versions, 'native_sha256': native_hash,
                  'imports': {name: str(path) for name, path in paths.items()},
                  'distribution': dist.metadata['Name'], 'version': dist.version}, indent=2))
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect_wheel(wheel: Path) -> dict:
    verify_wheel_contents(wheel)
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        (metadata_name,) = (name for name in names if name.endswith(".dist-info/METADATA"))
        (wheel_name,) = (name for name in names if name.endswith(".dist-info/WHEEL"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        wheel_metadata = Parser().parsestr(archive.read(wheel_name).decode())
        tags = wheel_metadata.get_all("Tag", [])
        require(
            bool(tags) and all(tag.startswith("cp310-abi3-") for tag in tags),
            f"expected a genuine cp310 ABI3 wheel, got {tags}",
        )
        require(
            sorted(part.strip() for part in metadata["Requires-Python"].split(",")) == ["<3.16", ">=3.10"],
            "unexpected Python support metadata",
        )
        require("polars" in metadata.get_all("Provides-Extra", []), "wheel omits the Polars extra")
        polars_requirements = [item for item in metadata.get_all("Requires-Dist", []) if item.startswith("polars")]
        require(
            len(polars_requirements) == 1
            and polars_requirements[0].replace(" ", "").replace("'", '"') == 'polars==1.44.2;extra=="polars"',
            f"Polars must only be the tested optional extra, got {polars_requirements}",
        )
        native_sha = hashlib.sha256(archive.read("gr_tdigest/_gr_tdigest.abi3.so")).hexdigest()
    return {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "native_sha256": native_sha,
        "distribution": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist"),
        "tags": tags,
    }


def check_results(path: Path, expected: int) -> int:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    require(len(cases) == expected, f"expected {expected} cases in {path}; got {len(cases)}")
    require(
        not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")),
        f"skipped or failing cases in {path}",
    )
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="test this existing wheel; never retag or rebuild it")
    parser.add_argument("--sdist", type=Path, help="also check this prebuilt source archive with --wheel")
    parser.add_argument("--output-dir", type=Path, help="new directory for persistent logs and results")
    args = parser.parse_args()
    if args.sdist and not args.wheel:
        parser.error("--sdist requires --wheel; otherwise the gate builds both artifacts")
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="tdigest-support-results-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTEST_ADDOPTS", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        }
    )
    journal = []

    def run(label: str, command: list[str], cwd: Path, *, build: bool = False) -> str:
        print(f"{label} (log: {output / (label + '.log')})", flush=True)
        command_env = env.copy()
        if build:
            # The project tools already exist in the development environment;
            # these are never used by the isolated installed-wheel test cells.
            command_env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
            # Reuse dependency artifacts when compiling the verified archive
            # outside the checkout. Cargo still rebuilds the extracted sources.
            command_env["CARGO_TARGET_DIR"] = str((root / env.get("CARGO_TARGET_DIR", "target")).resolve())
        result = subprocess.run(
            command, cwd=cwd, env=command_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (output / (label + ".log")).write_text(result.stdout)
        journal.append({"label": label, "command": command, "cwd": str(cwd), "exit_code": result.returncode})
        (output / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        require(result.returncode == 0, f"{label} failed; see {output / (label + '.log')}")
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="tdigest-support-cells-") as temporary:
        work = Path(temporary)
        tests, parity = stage_compatibility_tests(root, work)
        modern = work / "modern"
        modern.mkdir()
        for relative in MODERN_TESTS:
            shutil.copyfile(root / relative, modern / Path(relative).name)
        run("python-runtimes", ["uv", "python", "install", *dict.fromkeys(row[0] for row in MATRIX)], work)
        cells = []
        variants = [(minor, numpy, scipy, pytest, "1.44.2") for minor, numpy, scipy, pytest in MATRIX]
        baselines = {}
        for minor, numpy, scipy, pytest in MATRIX:
            baselines.setdefault(minor, (minor, numpy, scipy, pytest, None))
        variants.extend(baselines.values())
        for minor, numpy, scipy, pytest, polars in variants:
            label = f"py{minor.replace('.', '')}-numpy{numpy.split('.')[0]}" + ("-no-polars" if polars is None else "")
            venv = work / label
            run(label + "-venv", ["uv", "venv", "--python", minor, str(venv)], work)
            python = str(venv / "bin/python")
            run(label + "-pip", [python, "-I", "-m", "ensurepip", "--upgrade"], work)
            requirements = [f"numpy=={numpy}", f"scipy=={scipy}", f"pytest=={pytest}"]
            if polars:
                requirements.append(f"polars=={polars}")
            run(
                label + "-deps",
                ["uv", "pip", "install", "--only-binary", ":all:", "--python", python, *requirements],
                work,
            )
            cells.append(
                {
                    "label": label,
                    "python": minor,
                    "executable": python,
                    "numpy": numpy,
                    "scipy": scipy,
                    "polars": polars,
                    "pytest": pytest,
                }
            )
        if args.wheel:
            wheel = args.wheel.resolve(strict=True)
            sdist = args.sdist.resolve(strict=True) if args.sdist else None
            if sdist:
                verify_sdist_contents(sdist, root)
        else:
            wheels = output / "wheels"
            sdist = build_and_check_sdist(
                root, output / "sdist", lambda *command, cwd: run("source-archive", list(command), cwd, build=True)
            )
            unpack = work / "source"
            unpack.mkdir()
            with tarfile.open(sdist, "r:gz") as archive:
                # Verification rejected traversal, links and duplicate paths.
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(unpack, filter="data")
                else:
                    archive.extractall(unpack)
            (source,) = unpack.iterdir()
            # Compile the archive once on the oldest supported interpreter.
            # Every runtime cell below checks this same ABI3 native binary.
            run(
                "release-wheel",
                [
                    "uv",
                    "run",
                    "--project",
                    str(root / "bindings/python"),
                    "--no-sync",
                    "maturin",
                    "build",
                    "--release",
                    "--locked",
                    "-F",
                    "python",
                    "--compatibility",
                    "linux",
                    "--interpreter",
                    cells[0]["executable"],
                    "-o",
                    str(wheels),
                ],
                source,
                build=True,
            )
            (wheel,) = wheels.glob("*.whl")
        artifact = inspect_wheel(wheel)
        if sdist:
            artifact.update({"sdist": str(sdist), "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest()})
        if not args.wheel:
            artifact["source_build_python"] = cells[0]["python"]
        (output / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
        results = []
        for cell in cells:
            label, python = cell["label"], cell["executable"]
            result = {**cell, "status": "RUNNING"}
            results.append(result)
            try:
                # Dependencies are already present. Keep normal pip dependency
                # checking enabled: the base cell must not request Polars, and
                # the extra cells must resolve the advertised Polars extra.
                requirement = str(wheel) + ("[polars]" if cell["polars"] else "")
                run(label + "-install", [python, "-I", "-m", "pip", "install", "--no-index", requirement], work)
                run(label + "-pip-check", [python, "-I", "-m", "pip", "check"], work)
                imports = run(label + "-imports", [python, "-I", "-c", INSPECT, json.dumps({**artifact, **cell})], work)
                result["installed"] = json.loads(imports)
                suites = [("compatibility", [tests, parity], 208)]
                if cell["polars"]:
                    suites.append(("modern", [modern], 184))
                else:
                    suites.append(
                        (
                            "modern_python",
                            [
                                modern / name
                                for name in (
                                    "test_api_python.py",
                                    "test_day_numpy_input.py",
                                    "test_day_numpy_output.py",
                                    "test_digest_regressions.py",
                                )
                            ],
                            120,
                        )
                    )
                for suite, directories, expected in suites:
                    xml = output / f"{label}-{suite}.xml"
                    run(
                        label + "-" + suite,
                        [
                            python,
                            "-I",
                            "-m",
                            "pytest",
                            "-q",
                            "-c",
                            "/dev/null",
                            "-p",
                            "no:cacheprovider",
                            "-o",
                            "xfail_strict=true",
                            "--junitxml",
                            str(xml),
                            *(str(path) for path in directories),
                        ],
                        work,
                    )
                    result[suite + "_passed"] = check_results(xml, expected)
                result["status"] = "PASS"
            except Exception as error:
                result["status"] = "FAIL"
                result["error"] = str(error)
            (output / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        require(hashlib.sha256(wheel.read_bytes()).hexdigest() == artifact["wheel_sha256"], "wheel changed during gate")
        require(
            len(results) == len(variants) and all(result["status"] == "PASS" for result in results),
            f"runtime support failed; inspect {output}",
        )
    print(
        f"{len(MATRIX)} full cells passed compatibility and modern checks; "
        f"{len(baselines)} Polars-free cells passed; evidence: {output}"
    )
    if not args.wheel:
        print(f"Source archive compiled once with Python {cells[0]['python']}; wheel passed every runtime cell")


if __name__ == "__main__":
    main()
