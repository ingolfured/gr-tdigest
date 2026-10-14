"""Fail if the pinned numerical implementation or public wrapper has changed."""

import hashlib
from pathlib import Path
import re


def verify_sources(root: Path) -> None:
    provenance = root / "crates/tdigest_core/PROVENANCE.md"
    checksums = re.findall(
        r"^([0-9a-f]{64})  (.+)$", provenance.read_text(), re.MULTILINE
    )
    if len(checksums) != 6:
        raise RuntimeError(
            "expected all four numerical source files, license, and Python wrapper"
        )
    for expected, relative in checksums:
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"original upstream implementation changed: {relative}")
    print("Original numerical source and Python wrapper checksums match.", flush=True)


if __name__ == "__main__":
    verify_sources(Path(__file__).resolve().parents[1])
