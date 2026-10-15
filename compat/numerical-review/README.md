# Compact numerical review evidence

Two independently scoped records are included:

- `original-rust/`: original committed-lock oracle and restored Rust API,
  crate packaging, external-consumer checks at commit 3b8fa69.
- `separated-modes/`: known compression/interpolation accuracy limitation at
  source 52f6212, with reduced examples. This is not a passing accuracy gate.

All material was copied/prepared in scratch while production source was
frozen. No binaries, Cargo target cache, source archives or huge exports are
included. Historical machine paths are labeled. Portable scripts are distinct
from historical executed scripts, with all hashes in `MANIFEST.json`.

The source/native/lock identities differ across the two scopes and must not be
collapsed into a claim about the currently building runtime candidate.
