"""Prepare by default; --run --cpu-slot-released builds/tests the actual sdist."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import tarfile
import time
import tomllib
import xml.etree.ElementTree as ET
from zipfile import ZipFile


COMMIT = '52f6212008111b62fa07374278b850868b9ea77e'
ARCHIVE_SHA = '7460857acbf9ab925e0136ee65be43d1c9edb12efaebd75beda0f485fd196a9d'
BASE = Path('/var/tmp/codex-tdigest-pr3-20261014')
REPO = BASE / 'work/gr-tdigest-f64-weights'
ARCHIVE = BASE / 'overnight/package-hygiene/validation-52f6212/dist/gr_tdigest-0.2.4.tar.gz'
WORK = BASE / 'overnight/package-hygiene/standard-sdist-wheel-52f6212'
SOURCE = WORK / 'extracted/gr_tdigest-0.2.4'
TARGET = BASE / 'work/gr-tdigest/target'
PYTHON = REPO / '.venv/bin/python'
MATURIN = REPO / '.venv/bin/maturin'
UV = Path('/home/ingo/.local/bin/uv')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(SOURCE)): sha(path)
        for path in sorted(SOURCE.rglob('*'))
        if path.is_file()
    }


def prepare() -> dict:
    assert sha(ARCHIVE) == ARCHIVE_SHA
    WORK.mkdir(parents=True, exist_ok=True)
    plan_path = WORK / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['archive_sha256'] == ARCHIVE_SHA
        for relative, expected in plan['source_file_sha256'].items():
            assert sha(SOURCE / relative) == expected, relative
        return plan

    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        if SOURCE.exists():
            # Preparation can resume after an assertion; never overwrite a
            # modified source tree or mix it with a different archive.
            for member in archive.getmembers():
                if member.isfile():
                    content = archive.extractfile(member)
                    assert content is not None
                    assert (WORK / 'extracted' / member.name).read_bytes() == content.read(), member.name
        else:
            (WORK / 'extracted').mkdir()
            archive.extractall(WORK / 'extracted', filter='data')
    project = tomllib.loads((SOURCE / 'pyproject.toml').read_text())
    assert project['project']['name'] == 'gr-tdigest'
    assert project['project']['readme'] == 'PYTHON_README.md'
    assert project['tool']['maturin']['manifest-path'] == 'Cargo.toml'
    assert (SOURCE / 'Cargo.toml').is_file()
    assert (SOURCE / 'Cargo.lock').is_file()
    assert (SOURCE / 'uv.lock').is_file()
    assert (SOURCE / 'bindings/python/uv.lock').is_file()
    checker = runpy.run_path(str(SOURCE / 'scripts/check_upstream_sources.py'))
    checker['verify_sources'](SOURCE)
    upstream = SOURCE / 'compat/tdigest-rs-upstream'
    checksums = re.findall(r'^([0-9a-f]{64})  (.+)$', (upstream / 'PROVENANCE.md').read_text(), re.MULTILINE)
    assert len(checksums) == 7
    for expected, relative in checksums:
        assert sha(upstream / relative) == expected, relative

    tests = WORK / 'tests'
    shutil.copytree(
        upstream / 'bindings/python/tests', tests,
        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'),
    )
    parity = WORK / 'parity'
    (parity / 'data').mkdir(parents=True)
    for name in (
        'test_original_numerics.py', 'test_legacy_compatibility.py',
        'test_optional_polars_import.py', 'test_published_pickle_migration.py',
    ):
        shutil.copyfile(SOURCE / 'bindings/python/tests' / name, parity / name)
    for name in ('upstream_numerical_golden.json', 'published_0_1_0_pickle_golden.json'):
        shutil.copyfile(SOURCE / 'bindings/python/tests/data' / name, parity / 'data' / name)
    (WORK / 'logs').mkdir()
    plan = {
        'status': 'prepared; no native build, installation, or test execution yet',
        'source_commit': COMMIT,
        'archive': str(ARCHIVE),
        'archive_sha256': ARCHIVE_SHA,
        'source_directory': str(SOURCE),
        'source_file_sha256': source_hashes(),
        'normalized_metadata_unchanged': True,
        'distribution_name': project['project']['name'],
        'readme': project['project']['readme'],
        'manifest_path': project['tool']['maturin']['manifest-path'],
        'pinned_original_sources_verified': True,
        'unchanged_upstream_test_file_checksums_verified': 7,
        'shared_target_dir': str(TARGET),
        'build_command': [str(MATURIN), 'build', '--release', '--locked', '-F', 'python', '--compatibility', 'linux', '-i', str(PYTHON), '-o', str(WORK / 'wheels')],
        'build_cwd': str(SOURCE),
        'expected_compatibility_tests': 208,
        'runtime_isolation': 'Fresh venv; python -I; cwd outside extracted source and repository; assert imports within venv and installed native hash equals wheel native hash.',
        'dependency_policy': 'Use the sdist Cargo.lock with --locked and the canonical bindings/python/uv.lock with uv export --project bindings/python --frozen; install only exported pinned dependencies plus the rebuilt wheel.',
        'lockfile_note': 'The sdist also includes the older repository-root uv.lock. It differs from the canonical binding lockfile (including missing scipy and different dev-tool versions). Leave both archive files unmodified and export compatibility-test dependencies from bindings/python exactly as the repository gate does. This does not alter the root metadata used for the wheel build.',
        'build_slot': 'Do not run until root explicitly releases CPU/cache; run mode requires --cpu-slot-released.',
        'sdist_probe_note': 'The original sdist invocation ran below the Rust crate, in bindings/python, which has no Cargo.toml. Maturin continued using configured ../../Cargo.toml and succeeded. The extracted normalized project has its own root Cargo.toml. Treat the preflight stderr as non-blocking unless this real rebuild fails.',
    }
    write_json(plan_path, plan)
    return plan


def execute(plan: dict) -> None:
    assert not (WORK / 'execution.json').exists(), 'Refusing to overwrite prior execution evidence'
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTEST_ADDOPTS', 'UV_PROJECT_ENVIRONMENT', 'VIRTUAL_ENV', 'PYO3_PYTHON', 'CARGO_TARGET_DIR'):
        env.pop(name, None)
    env.update(
        PATH='/home/ingo/.local/bin:/home/ingo/.cargo/bin:' + env['PATH'],
        CARGO_TARGET_DIR=str(TARGET), CARGO_BUILD_JOBS='2',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
        UV_CACHE_DIR='/home/ingo/.cache/uv', TERM='dumb',
    )
    record = {
        'source_commit': COMMIT, 'archive_sha256': ARCHIVE_SHA,
        'status': 'running', 'commands': [], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'shared_target_dir': str(TARGET), 'build_jobs': 2,
    }
    write_json(WORK / 'execution.json', record)

    def run(label: str, command: list[str], cwd: Path = WORK) -> str:
        started = time.monotonic()
        print('START ' + label + ': ' + repr(command), flush=True)
        log = WORK / 'logs' / (label + '.log')
        with log.open('w') as output:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        item = {'label': label, 'command': command, 'cwd': str(cwd), 'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started, 'log': str(log)}
        record['commands'].append(item)
        write_json(WORK / 'execution.json', record)
        if result.returncode:
            print(log.read_text(errors='replace')[-18000:], flush=True)
            raise subprocess.CalledProcessError(result.returncode, command)
        print(json.dumps(item), flush=True)
        return log.read_text()

    try:
        record['maturin_version'] = run('maturin-version', [str(MATURIN), '--version']).strip()
        record['rustc_version'] = run('rustc-version', ['rustc', '-Vv']).strip()
        record['python_version'] = run('python-version', [str(PYTHON), '-V']).strip()
        run('build-wheel', plan['build_command'], SOURCE)
        (wheel,) = (WORK / 'wheels').glob('*.whl')
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            bad = [name for name in names if '__pycache__' in PurePosixPath(name).parts or PurePosixPath(name).suffix in {'.pyc', '.pyo'}]
            assert not bad, bad
            (metadata_name,) = [name for name in names if name.endswith('.dist-info/METADATA')]
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            assert metadata['Name'] == 'gr-tdigest'
            natives = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names if name.endswith('.so')}
            assert len(natives) == 1 and 'gr_tdigest/_gr_tdigest.abi3.so' in natives
        record.update(wheel=str(wheel), wheel_sha256=sha(wheel), native_sha256=natives, archive_members=names, forbidden_members=bad)
        requirements = WORK / 'requirements.txt'
        run('export-dependencies', [str(UV), 'export', '--project', str(SOURCE / 'bindings/python'), '--frozen', '--all-groups', '--no-emit-project', '--no-hashes', '--output-file', str(requirements)])
        record['requirements_sha256'] = sha(requirements)
        venv = WORK / 'venv'
        run('create-venv', [str(UV), 'venv', '--python', str(PYTHON), str(venv)])
        python = str(venv / 'bin/python')
        run('install-wheel', [str(UV), 'pip', 'install', '--python', python, '-r', str(requirements), str(wheel)])
        imported = run('isolated-imports', [python, '-I', '-c',
            'import hashlib,json,pathlib,sys,tdigest_rs,gr_tdigest,gr_tdigest._gr_tdigest as native; '
            'prefix=pathlib.Path(sys.prefix).resolve(); '
            'paths={"legacy":pathlib.Path(tdigest_rs.__file__).resolve(),"modern":pathlib.Path(gr_tdigest.__file__).resolve(),"native":pathlib.Path(native.__file__).resolve()}; '
            'assert all(p.is_relative_to(prefix) for p in paths.values()); '
            'print(json.dumps({"paths":{k:str(v) for k,v in paths.items()},"native_sha256":hashlib.sha256(paths["native"].read_bytes()).hexdigest(),"python":sys.version}))'])
        imports = json.loads(imported.strip().splitlines()[-1])
        assert imports['native_sha256'] == natives['gr_tdigest/_gr_tdigest.abi3.so']
        record['isolated_imports'] = imports
        junit = WORK / 'compatibility.junit.xml'
        run('compatibility-tests', [python, '-I', '-m', 'pytest', '-q', str(WORK / 'tests'), str(WORK / 'parity'), '--junitxml=' + str(junit)])
        suites = list(ET.parse(junit).getroot().iter('testsuite'))
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
        assert counts == {'tests': 208, 'failures': 0, 'errors': 0, 'skipped': 0}, counts
        record['compatibility_results'] = counts
        changed = [name for name, expected in plan['source_file_sha256'].items() if sha(SOURCE / name) != expected]
        assert not changed, changed
        record['extracted_source_files_unchanged_after_build'] = True
        record['status'] = 'passed'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        raise
    finally:
        record['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(WORK / 'execution.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cpu-slot-released', action='store_true')
    args = parser.parse_args()
    if args.run and not args.cpu_slot_released:
        parser.error('--run requires explicit --cpu-slot-released after coordination')
    plan = prepare()
    if args.run:
        execute(plan)
    else:
        print(json.dumps({'status': plan['status'], 'plan': str(WORK / 'plan.json'), 'build_command': plan['build_command'], 'source_file_count': len(plan['source_file_sha256'])}, indent=2))
