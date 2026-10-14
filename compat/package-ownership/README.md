# Package ownership evidence

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
