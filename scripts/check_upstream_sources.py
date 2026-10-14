"""Fail if the pinned numerical implementation or public wrapper has changed."""

import hashlib
from pathlib import Path
import re


def verify_sources(root: Path) -> None:
    provenance = root / "src/upstream/PROVENANCE.md"
    checksums = re.findall(
        r"^([0-9a-f]{64})  (.+)$", provenance.read_text(), re.MULTILINE
    )
    if len(checksums) != 6:
        raise RuntimeError(
            "expected all four numerical source files, license, and Python wrapper"
        )
    namespace_imports = {
        "src/upstream/lib.rs": (b"use self::{", b"use crate::{"),
        "src/upstream/core.rs": (b"use super::{", b"use crate::{"),
        "src/upstream/scale.rs": (
            b"use super::traits::FloatConst;",
            b"use crate::traits::FloatConst;",
        ),
    }
    for expected, relative in checksums:
        content = (root / relative).read_bytes()
        if relative in namespace_imports:
            adapted, original = namespace_imports[relative]
            if content.count(adapted) != 1:
                raise RuntimeError(f"unexpected namespace adaptation: {relative}")
            content = content.replace(adapted, original, 1)
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"original upstream implementation changed: {relative}")
    print("Original numerical source and Python wrapper checksums match.", flush=True)


if __name__ == "__main__":
    verify_sources(Path(__file__).resolve().parents[1])
