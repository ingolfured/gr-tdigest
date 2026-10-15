"""Prepare narrow LF checkout patches; run real checksum guards in Linux scratch."""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


BASE = Path('/var/tmp/codex-tdigest-pr3-20261014')
SOURCES = {
    'private': (BASE / 'work/gr-tdigest', '8fff555623ca7a472ac38bc9066dbb81534e424d', '### Fixed\n'),
    'upstream': (BASE / 'work/tdigest-rs-upstream38', 'bc1f93e15c216b14b532bb005a2a2e2e662a08f8', '### Compatibility-preserving refresh\n'),
}
GUARD_PATHS = [
    'scripts/check_upstream_sources.py',
    'scripts/test_legacy_wheel.py',
    'scripts/check_python_sdist.py',
]
PROVENANCE_PATHS = ['src/upstream/PROVENANCE.md', 'compat/tdigest-rs-upstream/PROVENANCE.md']
PINNED_PATHS = {
    'src/upstream/LICENSE', 'bindings/python/tdigest_rs/lib.py',
    'src/upstream/core.rs', 'src/upstream/lib.rs', 'src/upstream/scale.rs', 'src/upstream/traits.rs',
    'compat/tdigest-rs-upstream/bindings/python/tests/__init__.py',
    'compat/tdigest-rs-upstream/bindings/python/tests/conftest.py',
    'compat/tdigest-rs-upstream/bindings/python/tests/constants.py',
    'compat/tdigest-rs-upstream/bindings/python/tests/test_distributions.py',
    'compat/tdigest-rs-upstream/bindings/python/tests/test_tdigest.py',
    'compat/tdigest-rs-upstream/bindings/python/tests/utils.py',
    'compat/tdigest-rs-upstream/bindings/python/benchmarks/run.py',
}
CI_LOGS = {
    '116196989529': 'ae3994fb22ccf6c205446caa6e2bb43d52fd30f7b62568e978f4b5361b816bb8',
    '116196989691': '992926cac8faef4b684d56930ce7ce5c752392f2ba27ed712afc283f39fe29a3',
    '116196989773': 'e1fd23c8d4619b968537bba81128213cdd201bb4f56e096960f217980ce08339',
}
GUARD_CODE = (
    'import sys; from pathlib import Path; '
    'root=Path(sys.argv[1]); sys.path.insert(0,str(root/"scripts")); '
    'from test_legacy_wheel import verify_upstream_snapshot; verify_upstream_snapshot(root)'
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f'{args}: {result.returncode}\n{result.stdout.decode(errors="replace")}\n{result.stderr.decode(errors="replace")}')
    return result


def git(root: Path, *args: str) -> bytes:
    return run(['git', '-C', str(root), *args]).stdout


def blob(root: Path, head: str, name: str) -> bytes:
    return git(root, 'show', f'{head}:{name}')


def put(root: Path, relative: str, content: bytes) -> None:
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


def pins(root: Path, head: str) -> dict[str, str]:
    result = {}
    for i, relative in enumerate(PROVENANCE_PATHS):
        entries = re.findall(r'^([0-9a-f]{64})  (.+)$', blob(root, head, relative).decode(), re.MULTILINE)
        assert len(entries) == (6 if i == 0 else 7)
        for expected, name in entries:
            full = name if i == 0 else 'compat/tdigest-rs-upstream/' + name
            assert full not in result
            result[full] = expected
    assert result.keys() == PINNED_PATHS
    return result


def guard(root: Path, log: Path, expected: int) -> dict:
    command = [sys.executable, '-I', '-B', '-c', GUARD_CODE, str(root)]
    result = run(command, check=False)
    log.write_bytes(result.stdout + result.stderr)
    assert result.returncode == expected, log.read_text()
    if expected:
        assert b'original upstream implementation changed: src/upstream/LICENSE' in result.stderr
    else:
        assert b'All seven upstream file checksums match the pinned snapshot.' in result.stdout
    return {'command': command, 'returncode': result.returncode, 'log': log.name, 'sha256': sha(log.read_bytes())}


def byte_inventory(root: Path, paths: list[str]) -> dict:
    return {p: {'sha256': sha((root / p).read_bytes()), 'bytes': (root / p).stat().st_size,
                'crlf': (root / p).read_bytes().count(b'\r\n'), 'lf': (root / p).read_bytes().count(b'\n')}
            for p in sorted(paths)}


def process(label: str, root: Path, head: str, marker: str, output: Path) -> dict:
    assert git(root, 'rev-parse', 'HEAD').decode().strip() == head
    before_status = git(root, 'status', '--porcelain=v1', '--untracked-files=all')
    assert before_status == b'', before_status
    before_tree = git(root, 'ls-tree', '-rz', head)
    directory = output / label
    directory.mkdir()
    expected = pins(root, head)
    attrs = blob(root, head, '.gitattributes')
    entries = ('\n# Preserve the exact LF bytes verified by the upstream checksum guards.\n' +
               ''.join(f'{p} text eol=lf\n' for p in sorted(expected))).encode()
    assert all(f'{p} text eol=lf'.encode() not in attrs for p in expected)
    newattrs = attrs.rstrip(b'\n') + b'\n' + entries
    changelog = blob(root, head, 'CHANGELOG.md')
    note = ('- Preserve LF checkout bytes for the 13 checksum-bound original implementation, Python wrapper, tests and benchmark files, including when Git uses `core.autocrlf=true`; keep the raw-byte checks and numerical code unchanged.\n')
    assert marker in changelog.decode()
    newchangelog = changelog.decode().replace(marker, marker + note, 1).encode()
    patch = ''.join(
        ''.join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
                                     fromfile='a/' + name, tofile='b/' + name))
        for name, old, new in [('.gitattributes', attrs, newattrs), ('CHANGELOG.md', changelog, newchangelog)]
    ).encode()
    patch_path = directory / 'checkout-lf.patch'
    patch_path.write_bytes(patch)
    git(root, 'apply', '--check', str(patch_path))
    changed_paths = git(root, 'apply', '--numstat', str(patch_path)).decode()
    assert {line.split('\t')[-1] for line in changed_paths.splitlines()} == {'.gitattributes', 'CHANGELOG.md'}
    repo = directory / 'scratch-index'
    repo.mkdir()
    run(['git', 'init', '--quiet', str(repo)])
    git(repo, 'config', 'core.autocrlf', 'false')
    git(repo, 'config', 'core.safecrlf', 'false')
    copied = sorted(set(expected) | set(PROVENANCE_PATHS) | set(GUARD_PATHS) | {'.gitattributes', 'CHANGELOG.md'})
    for name in copied:
        put(repo, name, blob(root, head, name))
    # Explicitly unpinned controls prove that the policy is not a blanket conversion.
    put(repo, 'checkout-controls/unpinned-lf.txt', b'one\ntwo\n')
    put(repo, 'checkout-controls/unpinned-crlf.csv', b'column,value\r\nfirst,1\r\n')
    controls = ['checkout-controls/unpinned-lf.txt', 'checkout-controls/unpinned-crlf.csv']
    real_csv = 'compat/accuracy/matched-budgets-ff73cfa/results/policy-outcomes.csv'
    if run(['git', '-C', str(root), 'cat-file', '-e', f'{head}:{real_csv}'], check=False).returncode == 0:
        put(repo, real_csv, blob(root, head, real_csv))
        controls.append(real_csv)
    git(repo, 'add', '--all')
    index_before = git(repo, 'ls-files', '-s').decode()
    normal = guard(repo, directory / '01-original-lf-guard.log', 0)
    negative = directory / 'negative-autocrlf'
    negative.mkdir()
    git(repo, '-c', 'core.autocrlf=true', 'checkout-index', '--all', '--force', f'--prefix={negative}/')
    negative_result = guard(negative, directory / '02-autocrlf-negative-guard.log', 1)
    negative_bytes = byte_inventory(negative, copied + controls)
    assert negative_bytes['src/upstream/LICENSE']['crlf'] > 0
    put(repo, '.gitattributes', newattrs)
    put(repo, 'CHANGELOG.md', newchangelog)
    git(repo, 'add', '--', '.gitattributes', 'CHANGELOG.md')
    index_after = git(repo, 'ls-files', '-s').decode()
    before_entries = {line.split('\t')[1]: line.split('\t')[0] for line in index_before.splitlines()}
    after_entries = {line.split('\t')[1]: line.split('\t')[0] for line in index_after.splitlines()}
    assert {name for name in before_entries if before_entries[name] != after_entries[name]} == {'.gitattributes', 'CHANGELOG.md'}
    corrected = directory / 'corrected-autocrlf'
    corrected.mkdir()
    git(repo, '-c', 'core.autocrlf=true', 'checkout-index', '--all', '--force', f'--prefix={corrected}/')
    corrected_result = guard(corrected, directory / '03-corrected-autocrlf-guard.log', 0)
    corrected_bytes = byte_inventory(corrected, copied + controls)
    assert all((corrected / p).read_bytes() == blob(root, head, p) for p in expected)
    assert all((negative / p).read_bytes() == (corrected / p).read_bytes() for p in controls)
    assert (corrected / controls[0]).read_bytes() == b'one\r\ntwo\r\n'
    assert (corrected / controls[1]).read_bytes() == b'column,value\r\nfirst,1\r\n'
    if real_csv in controls:
        assert (corrected / real_csv).read_bytes() == blob(root, head, real_csv)
    license_path = corrected / 'src/upstream/LICENSE'
    original_license = license_path.read_bytes()
    license_path.write_bytes(original_license + b'deliberate negative control\n')
    mutation = guard(corrected, directory / '04-deliberate-mutation-negative-guard.log', 1)
    license_path.write_bytes(original_license)
    restored = guard(corrected, directory / '05-restored-corrected-guard.log', 0)
    after_status = git(root, 'status', '--porcelain=v1', '--untracked-files=all')
    after_tree = git(root, 'ls-tree', '-rz', head)
    assert after_status == before_status == b'' and after_tree == before_tree
    assert git(root, 'rev-parse', 'HEAD').decode().strip() == head
    (directory / 'source-tree.git-ls-tree-z').write_bytes(before_tree)
    return {
        'status': 'PASS', 'source_repository': str(root), 'source_head': head,
        'patch': str(patch_path), 'patch_sha256': sha(patch), 'patch_numstat': changed_paths,
        'guard_and_pin_git_blob_sha256': {p: sha(blob(root, head, p)) for p in GUARD_PATHS + PROVENANCE_PATHS},
        'original_pin_sha256': expected,
        'pinned_git_blob_sha256': {p: sha(blob(root, head, p)) for p in sorted(expected)},
        'controls': controls, 'original_lf_guard': normal, 'negative_autocrlf_guard': negative_result,
        'corrected_autocrlf_guard': corrected_result, 'mutation_negative_guard': mutation, 'restored_guard': restored,
        'negative_inventory': negative_bytes, 'corrected_inventory': corrected_bytes,
        'source_git_tree_listing_sha256_before': sha(before_tree),
        'source_git_tree_listing_sha256_after': sha(after_tree),
        'source_clean_before_and_after': True,
        'only_changed_scratch_index_paths': ['.gitattributes', 'CHANGELOG.md'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failure-logs', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux', 'This control is intentionally Linux-only.'
    assert not args.output.exists(), 'Preserve prior runs; choose a new scratch output.'
    args.output.mkdir(parents=True)
    summary = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'driver_sha256': sha(Path(__file__).read_bytes()), 'python': sys.version,
        'git_version': run(['git', '--version']).stdout.decode().strip(),
        'platform': sys.platform,
        'scope': 'Linux Git core.autocrlf checkout and actual raw-byte guards; not native Windows validation.',
        'repositories': {},
        'remote_failure_run': 'https://github.com/ingolfured/tdigest-rs/actions/runs/38665683258',
        'remote_failure_logs': {},
    }
    for job, expected in CI_LOGS.items():
        content = (args.failure_logs / f'{job}.raw.log').read_bytes()
        assert sha(content) == expected
        assert b'RuntimeError: original upstream implementation changed: src/upstream/LICENSE' in content
        put(args.output, f'ci-failure-logs/{job}.raw.log', content)
        summary['remote_failure_logs'][job] = {'sha256': expected, 'bytes': len(content),
            'url': f'https://github.com/ingolfured/tdigest-rs/actions/runs/38665683258/job/{job}'}
    for label, (root, head, marker) in SOURCES.items():
        summary['repositories'][label] = process(label, root, head, marker, args.output)
    assert summary['repositories']['private']['original_pin_sha256'] == summary['repositories']['upstream']['original_pin_sha256']
    summary['status'] = 'PASS'
    (args.output / 'RESULTS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'results': str(args.output / 'RESULTS.json'),
                      'results_sha256': sha((args.output / 'RESULTS.json').read_bytes()),
                      'patches': {label: result['patch_sha256'] for label, result in summary['repositories'].items()}}, indent=2))


if __name__ == '__main__':
    main()
