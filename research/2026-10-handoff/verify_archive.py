#!/usr/bin/env python3
"""Verify historical research bytes; optionally extract into a NEW directory.

This does not execute any archived binaries, tests, helpers or benchmarks.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buf):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            n = self.current.readinto(buf)
            if n:
                return n
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def verify(root, destination=None, check_refs=False):
    archive_dir = root / 'archives'
    summary = json.loads((root / 'archive-summary.json').read_text())
    manifest_path = archive_dir / 'MANIFEST.json.gz'
    assert sha_file(manifest_path) == summary['manifest_sha256'], 'Manifest hash mismatch'
    with gzip.open(manifest_path, 'rt') as f:
        manifest = json.load(f)
    assert manifest['schema'] == 'tdigest-research-archive-v2'
    expected = {f['path']: f for f in manifest['files']}
    assert len(expected) == len(manifest['files']), 'Duplicate manifest paths'
    if destination is not None:
        destination = destination.resolve()
        if destination.exists():
            raise FileExistsError('Extraction requires a new directory: ' + str(destination))
        destination.mkdir(parents=True)
    seen = set()
    total = 0
    for archive in manifest['archives']:
        parts = []
        for part in archive['parts']:
            assert PurePosixPath(part['path']).name == part['path']
            path = archive_dir / part['path']
            assert path.stat().st_size == part['bytes'], str(path)
            assert sha_file(path) == part['sha256'], str(path)
            parts.append(path)
        with io.BufferedReader(PartsReader(parts)) as source:
            with tarfile.open(fileobj=source, mode='r|gz') as tf:
                for member in tf:
                    pure = PurePosixPath(member.name)
                    assert member.isfile() and not pure.is_absolute() and '..' not in pure.parts
                    assert member.name in expected and member.name not in seen, member.name
                    record = expected[member.name]
                    assert record['archive'] == archive['name'] and member.size == record['bytes']
                    stream = tf.extractfile(member)
                    target = destination.joinpath(*pure.parts) if destination else None
                    out = None
                    if target is not None:
                        assert target.is_relative_to(destination)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        out = target.open('xb')
                    h = hashlib.sha256()
                    try:
                        while data := stream.read(1024 * 1024):
                            h.update(data)
                            if out:
                                out.write(data)
                    finally:
                        stream.close()
                        if out:
                            out.close()
                    assert h.hexdigest() == record['sha256'], member.name
                    if target is not None:
                        os.chmod(target, record.get('mode', 0o644) & 0o777)
                    seen.add(member.name)
                    total += member.size
        print(f"Verified {archive['name']}: {archive['file_count']} files", flush=True)
    assert seen == set(expected), 'Missing archive files'
    checked_refs = 0
    if check_refs:
        mapping = json.loads((root / 'branch-map.json').read_text())
        repo = root.parent.parent
        for entry in mapping['entries']:
            if entry['repository'] != 'ingolfured/gr-tdigest':
                continue
            ref = 'refs/remotes/origin/' + entry['remote_branch']
            sha = subprocess.check_output(['git', 'rev-parse', ref], cwd=repo, text=True).strip()
            assert sha == entry['commit'], ref
            checked_refs += 1
    result = {'status': 'verified', 'files': len(seen), 'uncompressed_bytes': total,
              'archives': len(manifest['archives']), 'checked_branch_mappings': checked_refs,
              'extracted_to': str(destination) if destination else None}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--check-refs', action='store_true', help='Check mapped origin refs after a normal full clone/fetch')
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.extract, args.check_refs)
