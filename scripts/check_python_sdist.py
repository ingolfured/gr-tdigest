"""Check a real Maturin source archive without extracting it."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development dependency
    import tomli as tomllib


def canonical_relative(name: str) -> str:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name.rstrip("/"):
        raise RuntimeError(f"noncanonical archive path: {name!r}")
    return str(path)


def verify_sdist_contents(sdist: Path, root: Path) -> None:
    """Require active source bytes and reject aliases, duplicates and caches."""
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [canonical_relative(member.name) for member in members]
        counts = Counter(names)
        duplicate = sorted(name for name, count in counts.items() if count != 1)
        if duplicate:
            raise RuntimeError("sdist repeats member paths: " + ", ".join(duplicate))
        forbidden = sorted(
            name
            for name in names
            if "__pycache__" in PurePosixPath(name).parts or PurePosixPath(name).suffix in {".pyc", ".pyo"}
        )
        if forbidden:
            raise RuntimeError("sdist contains Python caches: " + ", ".join(forbidden))
        top_levels = {PurePosixPath(name).parts[0] for name in names}
        if len(top_levels) != 1:
            raise RuntimeError("sdist must have exactly one top-level directory")
        prefix = top_levels.pop() + "/"
        hashes = {}
        manifests = {}
        for member, name in zip(members, names):
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"sdist contains unsupported link or special member: {name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read sdist member: {name}")
                content = stream.read()
                relative = name.removeprefix(prefix)
                hashes[relative] = hashlib.sha256(content).hexdigest()
                if PurePosixPath(relative).name in {"Cargo.toml", "pyproject.toml"}:
                    manifests[relative] = tomllib.loads(content.decode())
        for relative in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
            if relative not in hashes:
                raise RuntimeError(f"sdist missing normalized build input: {relative}")

        def require_same(archive_path: str, source_path: str) -> None:
            expected = hashlib.sha256((root / source_path).read_bytes()).hexdigest()
            if hashes.get(archive_path) != expected:
                raise RuntimeError(f"sdist missing or changed source: {archive_path}")

        for relative in (
            "README.md",
            "bindings/python/PYTHON_README.md",
            "bindings/python/gr_tdigest/__init__.py",
            "bindings/python/tdigest_rs/__init__.py",
            "bindings/python/tdigest_rs/lib.py",
            "bindings/python/tdigest_rs/tdigest_rs.py",
            "src/tdigest/tdigest.rs",
            "src/upstream_py.rs",
        ):
            require_same(relative, relative)
        require_same("PYTHON_README.md", "bindings/python/PYTHON_README.md")
        normalized_build = manifests["pyproject.toml"].get("tool", {}).get("maturin", {})
        if normalized_build.get("manifest-path") != "Cargo.toml":
            raise RuntimeError("sdist has an unexpected active Rust manifest")
        if normalized_build.get("python-source", ".") != ".":
            raise RuntimeError("sdist has an unexpected active Python source directory")
        for relative in (
            "gr_tdigest/__init__.py",
            "tdigest_rs/__init__.py",
            "tdigest_rs/lib.py",
            "tdigest_rs/tdigest_rs.py",
        ):
            require_same(relative, "bindings/python/" + relative)
        source_manifest = tomllib.loads((root / "Cargo.toml").read_text())
        source_library = canonical_relative(source_manifest.get("lib", {}).get("path", "src/lib.rs"))
        archive_library = canonical_relative(manifests["Cargo.toml"].get("lib", {}).get("path", "src/lib.rs"))
        if archive_library != source_library:
            raise RuntimeError("sdist changed the active root Rust library path")
        require_same(archive_library, source_library)
        for source in sorted((root / "src").rglob("*.rs")):
            if not source.is_relative_to(root / "src/upstream"):
                relative = source.relative_to(root).as_posix()
                require_same(relative, relative)
        if "tdigest_core" in source_manifest.get("dependencies", {}):
            dependency = manifests["Cargo.toml"].get("dependencies", {}).get("tdigest_core", {})
            if not isinstance(dependency, dict) or not isinstance(dependency.get("path"), str):
                raise RuntimeError("sdist missing the local tdigest_core dependency path")
            core_dir = canonical_relative(dependency["path"])
            core_manifest = manifests.get(core_dir + "/Cargo.toml", {})
            if core_manifest.get("package", {}).get("name") != "tdigest_core":
                raise RuntimeError("sdist active original core has the wrong package identity")
            if core_manifest.get("lib", {}).get("path") != "lib.rs":
                raise RuntimeError("sdist active original core has the wrong library path")
        else:
            core_dir = "src/upstream"  # Older snapshots used a private path module.
        for name in ("lib.rs", "core.rs", "scale.rs", "traits.rs", "LICENSE"):
            require_same(core_dir + "/" + name, "src/upstream/" + name)
    print("Source archive contains required active source bytes and no duplicate paths or Python caches.", flush=True)
