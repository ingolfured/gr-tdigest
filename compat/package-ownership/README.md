# Package ownership and real upgrade evidence

A real same-distribution upgrade prototype passed on Linux/Python 3.12 at
`52f6212`. A fresh immutable export changed only `[project].name` from
`gr-tdigest` to `tdigest-rs`, leaving Rust/native/module identities unchanged.
Maturin built the actual release wheel and sdist; wheel metadata was not repacked.
This experiment does not change this repository's production distribution name.

Using real pip, the probe installed the authentic published `tdigest-rs==0.1.0`,
upgraded to the prototype without manual removal, uninstalled it, then reinstalled
it. **All 32 old-restored states matched exactly**, and **208/208 compatibility
tests passed after both upgrade and reinstall**, without skips or xfails.
The old native extension was removed, the new shim selected the expected native
hash, `tdigest-rs` was the only namespace owner, and uninstall removed all owned
files/imports. Modern Python, wire and Polars smoke checks passed too.

Dependencies were staged explicitly: old NumPy 1.26.4, then NumPy 2.3.4 and Polars
1.34.0 for the candidate. pip ownership operations used `--no-deps`; this is not
an unconstrained resolver test. The environment contained no independent
`gr-tdigest` distribution, so co-installation remains unsupported. Python 3.10/3.11,
the original platform matrix, rollback constraints and ownership of a bundled
modern namespace remain release decisions.

The private artifact packaging also improved: explicit excludes remove Python
caches from wheels and sdists; the installed-wheel gate rejects them before
installation. A uniquely named Python README fixes Maturin's duplicate root
README path. Four injected cache markers were excluded from the real sdist.
That unmodified, normalized source archive was then rebuilt into a release wheel
and checked with the complete 208-case installed compatibility gate. This is
actual source rebuild evidence, not merely a source-file listing.

The [lifecycle result](recorded-52f6212/RESULT.json),
[artifact hashes](recorded-52f6212/artifacts.json),
[source export and exact name-only diff](recorded-52f6212/source-export.json),
[pip/build commands](recorded-52f6212/commands.json),
[standard sdist rebuild](recorded-52f6212/standard-sdist-rebuild.json), and
[record manifest](recorded-52f6212/manifest.json) preserve attribution.
The archived scripts retain the original WSL paths as experiment records; they
are not a general release command. Historical inspection below explains why a
same-distribution upgrade matters and remains valid for co-installation.

# Historical ownership inspection

The published `tdigest-rs==0.1.0` wheel and the preserved review `gr-tdigest==0.2.4` wheel both contain `tdigest_rs/__init__.py` and `tdigest_rs/lib.py`. Their distribution names differ, so installing the latter does not replace the former as an upgrade. Uninstalling either distribution can also remove shared files required by the other.

The original wheel contains `tdigest_rs/tdigest_rs.abi3.so`; the review wheel adds `tdigest_rs/tdigest_rs.py`, which forwards to the replacement native adapter. A Linux CPython `PathFinder` check with those two basenames selected the original `.abi3.so` with `ExtensionFileLoader`. Thus leaving both wheel contents installed can silently select the old implementation. The check used empty placeholders and inspected the import spec; it did not execute either module or simulate a full pip deployment.

`PROVENANCE.json` records the exact wheel hashes, archive paths, probe method and result. The first review wheel predates the single-centroid mean fix. Static ZIP inspection of the later wheel at `22867bd55739a96a6fab3c8c48b27406ecb32197` confirmed the same package ownership and filenames. Its SHA256 came from the preserved release gate; this audit did not repeat hashing or native tests during benchmark timing. The published wheel is version **0.1.0**; the separately pinned upstream source identifies itself as **0.2.0**.

Publish the upstream replacement under the same **`tdigest-rs` distribution name**, preserving the old `tdigest_rs.lib.TDigest` API. A normal same-distribution upgrade removes the old native extension before installing the replacement. Do not co-install the private compatibility wheel. Keep one distribution owner per namespace, and verify an actual old-wheel upgrade, selected native-module path, uninstall and reinstall before release. Bundling `gr_tdigest` into the upstream wheel would also require an ownership decision for independently installed `gr-tdigest`.

Both inspected review wheels also contain generated `__pycache__/*.pyc` files. Exclude Python caches explicitly in maturin packaging and assert their absence in the release wheel's ZIP entries. This is an artifact hygiene issue; no runtime failure is attributed to these files.

Reproduce the precedence check on Linux without loading a native module:

```python
from importlib.machinery import PathFinder
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    package = Path(directory)
    (package / "tdigest_rs.abi3.so").touch()
    (package / "tdigest_rs.py").touch()
    spec = PathFinder.find_spec("tdigest_rs.tdigest_rs", [str(package)])
    assert Path(spec.origin).name == "tdigest_rs.abi3.so"
    assert type(spec.loader).__name__ == "ExtensionFileLoader"
```
