"""Real pip same-distribution prototype probe. Build is an explicit separate phase."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from email.parser import Parser

BASE = Path('/var/tmp/codex-tdigest-pr3-20261014')
DEFAULT_ROOT = BASE / 'overnight/package-hygiene/same-distribution-probe'
DEFAULT_REPO = BASE / 'work/gr-tdigest-f64-weights'
PYTHON312 = '/home/ingo/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12'
UV = '/home/ingo/.local/bin/uv'
OLD_WHEEL = BASE / 'overnight/compatibility/published-pickle/tdigest_rs-0.1.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
OLD_SHA = '62ee8d5a01226d020a2ab11a956292c38beab3f34d2d9fa7a8fca22c3e5d4bbf'
OLD_NATIVE_SHA = 'e54bc0b860bbab8baff415e85796cbcdced013941bb763ed59958ccee6828866'
NEW_DEPENDENCIES = ['numpy==2.3.4', 'polars==1.34.0', 'scipy==1.17.1', 'pytest==8.4.2']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


class Runner:
    def __init__(self, root):
        self.root = root.resolve()
        require(self.root.is_relative_to(BASE / 'overnight/package-hygiene'), 'probe root must stay in dedicated scratch')
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'logs').mkdir(exist_ok=True)
        self.python = self.root / 'venv/bin/python'
        self.env = os.environ.copy()
        for key in ['PYTHONPATH', 'PYTEST_ADDOPTS', 'VIRTUAL_ENV']:
            self.env.pop(key, None)
        self.env.update({'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                         'PIP_NO_INPUT': '1', 'PIP_CONFIG_FILE': '/dev/null'})
        self.env['PATH'] = '/home/ingo/.cargo/bin:/home/ingo/.local/bin:/usr/sbin:' + self.env.get('PATH', '')
        self.journal = self.root / 'commands.json'
        self.commands = json.loads(self.journal.read_text()) if self.journal.exists() else []

    def run(self, phase, args, cwd=None, env=None):
        args = [str(x) for x in args]
        start = time.monotonic()
        result = subprocess.run(args, cwd=cwd or self.root, env=env or self.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        index = len(self.commands) + 1
        log = self.root / 'logs' / f'{index:03d}-{phase}.log'
        log.write_text(result.stdout)
        self.commands.append({'phase': phase, 'command': args, 'cwd': str(cwd or self.root),
                              'exit_code': result.returncode, 'seconds': time.monotonic() - start,
                              'log': str(log)})
        write(self.journal, self.commands)
        print(json.dumps({'phase': phase, 'exit_code': result.returncode, 'log': str(log)}), flush=True)
        if result.returncode:
            raise RuntimeError(f'{phase} failed; see {log}\n{result.stdout[-2000:]}')
        return result.stdout

    def pip(self, phase, *args):
        return self.run(phase, [self.python, '-I', '-m', 'pip', *args])

    def prepare(self):
        if not self.python.exists():
            self.run('create-venv', [UV, 'venv', '--offline', '--python', PYTHON312, self.root / 'venv'])
        self.run('bootstrap-pip', [self.python, '-I', '-m', 'ensurepip', '--upgrade'])
        version = self.pip('pip-version', '--version').strip()
        self.run('old-dependencies', [UV, 'pip', 'install', '--offline', '--python', self.python,
                                      'numpy==1.26.4', 'pytest==8.4.2'])
        require(sha(OLD_WHEEL) == OLD_SHA, 'authentic old wheel hash mismatch')
        write(self.root / 'preparation.json', {
            'python': PYTHON312, 'pip': version, 'old_wheel': str(OLD_WHEEL), 'old_wheel_sha256': OLD_SHA,
            'dependency_policy': 'Explicit staged NumPy1.26.4 -> NumPy2.3.4 transition; package upgrades/removals use real pip --no-deps.',
            'scope': 'Ownership/removal and tested outputs; not unconstrained resolver behavior or original Python support.',
            'native_build_performed': False,
            'bootstrap_note': 'Initial uv --seed --offline attempt found no cached pip; ensurepip supplied bundled pip25.0.1. No network bootstrap required.'
        })

    def export(self, repo, ref):
        require(bool(re.fullmatch('[0-9a-f]{40}', ref or '')), 'use an immutable full source commit')
        source_dir = self.root / 'same-name-source'
        require(not source_dir.exists(), 'source export already exists; preserve it and choose a fresh probe root')
        commit = self.run('resolve-source', ['git', '-C', repo, 'rev-parse', f'{ref}^{{commit}}']).strip()
        require(commit == ref, 'source ref mismatch')
        tree = self.run('source-tree', ['git', '-C', repo, 'rev-parse', f'{ref}^{{tree}}']).strip()
        archive = self.root / 'source.tar'
        self.run('archive-source', ['git', '-C', repo, 'archive', '--format=tar', '--output', archive, ref])
        source_dir.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(source_dir, filter='data')
        project_file = source_dir / 'bindings/python/pyproject.toml'
        original = project_file.read_text()
        parsed = tomllib.loads(original)
        require(parsed['project']['name'] == 'gr-tdigest', 'unexpected original project identity')
        exclusions = parsed['tool']['maturin'].get('exclude', [])
        require(all(x in exclusions for x in ['**/__pycache__/**', '**/*.pyc', '**/*.pyo']), 'final source must include hygiene exclusions')
        changed, count = re.subn(r'(?m)^name[ \t]*=[ \t]*"gr-tdigest"[ \t]*$', 'name = "tdigest-rs"', original)
        require(count == 1, 'project-name rewrite must be unique')
        expected = tomllib.loads(original)
        expected['project']['name'] = 'tdigest-rs'
        require(tomllib.loads(changed) == expected, 'temporary metadata changes extend beyond project.name')
        project_file.write_text(changed)
        diff = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                           fromfile='a/bindings/python/pyproject.toml', tofile='b/bindings/python/pyproject.toml'))
        (self.root / 'temporary-project-name.patch').write_text(diff)
        markers = [
            'bindings/python/gr_tdigest/__pycache__/codex_upgrade_probe.pyc',
            'bindings/python/tdigest_rs/codex_upgrade_probe.pyo',
            'bindings/python/tdigest_rs/__pycache__/nested/codex_upgrade_probe.txt',
            'codex_upgrade_root_probe.pyc',
        ]
        for relative in markers:
            path = source_dir / relative
            require(not path.exists(), f'cache marker collision: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'Harmless packaging exclusion probe; not Python bytecode.\n')
        write(self.root / 'source-export.json', {
            'source_commit': ref, 'source_tree': tree, 'source_repository': str(repo),
            'source_archive': str(archive), 'source_archive_sha256': sha(archive),
            'source_export': str(source_dir), 'prototype_label': 'Local same-distribution packaging prototype; not production release',
            'only_metadata_change': '[project].name: gr-tdigest -> tdigest-rs',
            'metadata_diff': diff,
            'original_pyproject_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'prototype_pyproject_sha256': sha(project_file),
            'cargo_package_unchanged': 'gr-tdigest', 'native_module_unchanged': parsed['tool']['maturin']['module-name'],
            'cache_markers': {relative: sha(source_dir / relative) for relative in markers},
        })
        self.stage_tests(source_dir)

    def stage_tests(self, source_dir):
        checks = self.root / 'checks'
        checks.mkdir()
        shutil.copytree(source_dir / 'compat/tdigest-rs-upstream/bindings/python/tests', checks / 'tests',
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        parity = checks / 'parity'
        (parity / 'data').mkdir(parents=True)
        names = ['test_original_numerics.py', 'test_legacy_compatibility.py',
                 'test_optional_polars_import.py', 'test_published_pickle_migration.py']
        for name in names:
            shutil.copyfile(source_dir / 'bindings/python/tests' / name, parity / name)
        for name in ['upstream_numerical_golden.json', 'published_0_1_0_pickle_golden.json']:
            shutil.copyfile(source_dir / 'bindings/python/tests/data' / name, parity / 'data' / name)
        shutil.copyfile(Path(__file__).with_name('probe_installed.py'), checks / 'probe_installed.py')
        write(self.root / 'staged-test-hashes.json', {
            str(path.relative_to(checks)): sha(path) for path in checks.rglob('*') if path.is_file()
        })

    def build(self, maturin, target_dir, authorized):
        require(authorized, 'native build requires explicit --build-authorized after root releases CPU/cache')
        require(self.python.exists(), 'run prepare first')
        provenance = json.loads((self.root / 'source-export.json').read_text())
        source_dir = Path(provenance['source_export'])
        artifacts = self.root / 'artifacts'
        require(not artifacts.exists(), 'preserve prior artifacts; choose a fresh root for a new build')
        artifacts.mkdir()
        env = self.env.copy()
        if target_dir:
            env['CARGO_TARGET_DIR'] = str(target_dir)
        self.run('maturin-version', [maturin, '--version'])
        self.run('rust-version', ['rustc', '-Vv'])
        self.run('cargo-version', ['cargo', '-V'])
        self.run('source-pin-check', [self.python, source_dir / 'scripts/check_upstream_sources.py'])
        self.run('build-prototype-wheel', [maturin, 'build', '--release', '-F', 'python', '--compatibility', 'linux',
                                          '--interpreter', self.python, '-o', artifacts],
                 cwd=source_dir / 'bindings/python', env=env)
        self.run('build-prototype-sdist', [maturin, 'sdist', '-o', artifacts],
                 cwd=source_dir / 'bindings/python', env=env)
        wheels = list(artifacts.glob('*.whl'))
        sdists = list(artifacts.glob('*.tar.gz'))
        require(len(wheels) == len(sdists) == 1, 'expected one actual wheel and sdist')
        wheel, sdist = wheels[0], sdists[0]
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.namelist()
            self.check_archive(entries)
            meta_file = next(x for x in entries if x.endswith('.dist-info/METADATA'))
            meta = Parser().parsestr(archive.read(meta_file).decode())
            require(re.sub(r'[-_.]+', '-', meta['Name']).lower() == 'tdigest-rs', 'wheel distribution identity')
            require(meta['Version'] == '0.2.4', 'prototype version differs; review upgrade ordering')
            require(meta_file.startswith('tdigest_rs-0.2.4.dist-info/'), 'dist-info ownership identity')
            native = 'gr_tdigest/_gr_tdigest.abi3.so'
            require(native in entries and 'tdigest_rs/tdigest_rs.py' in entries, 'expected native/shim layout')
            require('tdigest_rs/tdigest_rs.abi3.so' not in entries, 'prototype must not include original extension')
            native_sha = hashlib.sha256(archive.read(native)).hexdigest()
        with tarfile.open(sdist) as archive:
            members = archive.getnames()
            self.check_archive(members)
            for suffix in ['/Cargo.toml', '/src/lib.rs', '/bindings/python/pyproject.toml',
                           '/bindings/python/tdigest_rs/lib.py', '/bindings/python/gr_tdigest/__init__.py']:
                require(any(name.endswith(suffix) for name in members), f'sdist missing intended source: {suffix}')
        write(self.root / 'artifacts.json', {
            'source_commit': provenance['source_commit'], 'prototype': True,
            'wheel': str(wheel), 'wheel_sha256': sha(wheel), 'native_path': native, 'native_sha256': native_sha,
            'sdist': str(sdist), 'sdist_sha256': sha(sdist),
            'metadata': {'name': meta['Name'], 'version': meta['Version'], 'requires_python': meta['Requires-Python'],
                         'requires_dist': meta.get_all('Requires-Dist')},
            'archive_hygiene': 'Both actual artifacts exclude all injected cache entries and .pyc/.pyo paths.',
            'build_kind': 'Actual source export built by maturin; no wheel metadata repack.',
        })

    @staticmethod
    def check_archive(entries):
        bad = [x for x in entries if '__pycache__' in PurePosixPath(x).parts or PurePosixPath(x).suffix in ['.pyc', '.pyo']]
        require(not bad, f'forbidden generated archive entries: {bad}')

    def snapshot(self, phase, version=None, native_sha=None, baseline=None, previous=None):
        checks = self.root / 'checks'
        args = [self.python, '-I', checks / 'probe_installed.py', phase, '--output', self.root / f'{phase}.json']
        if phase != 'uninstalled':
            args += ['--fixture', checks / 'parity/data/published_0_1_0_pickle_golden.json',
                     '--expected-version', version, '--expected-native-sha256', native_sha]
        if baseline:
            args += ['--baseline', baseline]
        if previous:
            args += ['--previous-record', previous]
        self.run(f'inspect-{phase}', args, cwd=checks)

    def compatibility(self, phase):
        checks = self.root / 'checks'
        xml = self.root / f'{phase}-compatibility.xml'
        self.run(f'{phase}-208-compatibility', [self.python, '-I', '-m', 'pytest', '-q', '-c', '/dev/null',
                                              '-p', 'no:cacheprovider', '--junitxml', xml,
                                              checks / 'tests', checks / 'parity'], cwd=checks)
        testcases = list(ET.parse(xml).getroot().iter('testcase'))
        require(len(testcases) == 208, f'expected 208 cases; found {len(testcases)}')
        require(not any(c.find(tag) is not None for c in testcases for tag in ['failure', 'error', 'skipped']),
                'compatibility matrix contains skipped/failed cases')

    def probe(self):
        require(not (self.root / 'old.json').exists(), 'probe already started; preserve prior state and investigate rather than restart')
        artifacts = json.loads((self.root / 'artifacts.json').read_text())
        wheel = Path(artifacts['wheel'])
        require(sha(wheel) == artifacts['wheel_sha256'], 'prototype wheel changed')
        require(sha(OLD_WHEEL) == OLD_SHA, 'original wheel changed')
        self.pip('preflight-installed', 'list', '--format=json')
        self.run('require-clean-package-namespace', [self.python, '-I', '-c',
                 'import importlib.util; assert importlib.util.find_spec("tdigest_rs") is None; assert importlib.util.find_spec("gr_tdigest") is None'])
        self.pip('install-authentic-original', 'install', '--no-index', '--no-deps', str(OLD_WHEEL))
        self.pip('old-pip-check', 'check')
        self.snapshot('old', '0.1.0', OLD_NATIVE_SHA)
        # Preserve exact old-restored outputs before changing NumPy or package files.
        self.pip('old-dependency-freeze', 'freeze', '--all')
        self.run('explicit-dependency-transition', [UV, 'pip', 'install', '--offline', '--python', self.python,
                                                   *NEW_DEPENDENCIES])
        self.pip('upgrade-same-distribution', 'install', '--no-index', '--no-deps', '--upgrade', str(wheel))
        self.pip('upgraded-pip-check', 'check')
        self.pip('upgraded-dependency-freeze', 'freeze', '--all')
        self.snapshot('upgraded', artifacts['metadata']['version'], artifacts['native_sha256'], self.root / 'old.json')
        self.compatibility('upgraded')
        self.pip('uninstall-prototype', 'uninstall', '-y', 'tdigest-rs')
        self.snapshot('uninstalled', previous=self.root / 'upgraded.json')
        self.pip('reinstall-prototype', 'install', '--no-index', '--no-deps', str(wheel))
        self.pip('reinstalled-pip-check', 'check')
        self.snapshot('reinstalled', artifacts['metadata']['version'], artifacts['native_sha256'], self.root / 'old.json')
        self.compatibility('reinstalled')
        write(self.root / 'RESULT.json', {
            'status': 'PASS', 'scope': 'Real pip same-distribution ownership/upgrade/removal probe with explicit staged dependency transition.',
            'source_commit': artifacts['source_commit'], 'prototype_wheel_sha256': artifacts['wheel_sha256'],
            'old_wheel_sha256': OLD_SHA, 'pip_operations': 'Install old -> upgrade without manual uninstall -> uninstall -> reinstall.',
            'exact_old_restored_states': 32, 'compatibility_cases_after_upgrade': 208,
            'compatibility_cases_after_reinstall': 208, 'skipped_or_xfailed': 0,
            'old_native_removed': True, 'selected_new_native_hash_matches_artifact': True,
            'single_namespace_owner': 'tdigest-rs', 'uninstall_owned_files_removed': True,
            'modern_smoke': 'Python weighted median/wire and Polars plugin passed after upgrade and reinstall.',
            'limitations': ['Local packaging prototype, not production package identity policy.',
                           'Does not test unconstrained pip dependency resolution or original Python/platform support.',
                           'Clean initial environment has no independent gr-tdigest distribution; co-installation remains unsupported.',
                           'Old stage uses NumPy1.26.4; candidate stage uses NumPy2.3.4. Dependency transition is explicit.',
                           'No rollback claim. NumPy2 protocol5 -> NumPy1 namespace constraint remains separately documented.'],
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'export', 'build', 'probe'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--source-ref')
    parser.add_argument('--maturin', type=Path, default=DEFAULT_REPO / '.venv/bin/maturin')
    parser.add_argument('--cargo-target-dir', type=Path)
    parser.add_argument('--build-authorized', action='store_true')
    args = parser.parse_args()
    runner = Runner(args.root)
    if args.phase == 'prepare':
        runner.prepare()
    elif args.phase == 'export':
        runner.export(args.repo, args.source_ref)
    elif args.phase == 'build':
        runner.build(args.maturin, args.cargo_target_dir, args.build_authorized)
    else:
        runner.probe()


if __name__ == '__main__':
    main()
