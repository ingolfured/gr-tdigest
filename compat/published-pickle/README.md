# Published-package pickle migration

These 32 fixtures come from the actual PyPI `tdigest-rs` **0.1.0** Linux wheel,
installed with Python 3.12.12 and NumPy 1.26.4. They cover f32/f64, weighted data,
large uint32 weights, distinct construction/merge deltas, and pickle protocols 4/5.
The generator verifies the wheel SHA256 and checks that the installed wrapper and
native extension are byte-identical to its members. Inputs are synthetic.

The vendored source pin `8d9cf1fe82c5c9254e5dc9593f01717e317f4333` advertises
**0.2.0**; no published wheel for that version was found in the official PyPI and
GitHub release metadata when these fixtures were collected. The published 0.1.0
wrapper also lacks the later `update()` method. This migration gate complements
the pinned-source API and numerical tests; it does not replace them. Exact wheel,
fixture, generator, and native-extension hashes are in [PROVENANCE.json](PROVENANCE.json).

## Preserved behavior

Assertions compare the replacement with `pickle.loads` performed by the old
installed package: centroid bytes/dtypes, uint32 weights, length, median,
quantiles, and trimmed means. They do not assume pickling was lossless upstream.
The old state contains means and weights only; reconstruction uses default
`delta=300`. In the delta-5000 cases this already reduces 1833 centroids to 134
inside the old package. For example, its f64 median changes from `0.46875` to
`0.8301359097633136`. The replacement matches the old **restored** results exactly.
This inherited behavior must not be described as a new upgrade regression.

The forward migration passed all 32 cases in a clean installation of an optimized
candidate wheel with NumPy 2.3.4, outside the checkout with `python -I`; public and
native import paths were verified inside that environment. The recorded wheel
predates a separate modern singleton-mean fix. Its hash identifies the historical
artifact; the installed-wheel gate must be rerun for the final binary.

## Rollback boundary

Pickles re-saved by that candidate using protocol 4 restored exactly in the old
package with NumPy 1.26.4 (16/16 cases). Protocol 5 failed for all 16 cases with
`ModuleNotFoundError: No module named 'numpy._core.numeric'`: NumPy 2 writes a
module path absent in the tested NumPy 1 environment. This is a dependency/state
compatibility limit, not a numerical difference. Do not promise this rollback;
retain old state or explicitly use a tested interoperable serialization policy.
The existing original wrapper is unchanged.

## Reproduce and review

Download the original wheel linked in `PROVENANCE.json` and set `WHEEL` to its
local path. In an isolated Linux environment, generate only with the genuine old
package; never regenerate expected outputs from the replacement to accept a drift.

```sh
uv venv --python 3.12 /tmp/tdigest-pickle-original
uv pip install --python /tmp/tdigest-pickle-original/bin/python \
  --only-binary :all: numpy==1.26.4 "$WHEEL"
/tmp/tdigest-pickle-original/bin/python -I compat/published-pickle/generate.py \
  --wheel "$WHEEL" \
  --output bindings/python/tests/data/published_0_1_0_pickle_golden.json
```

`make test` collects the 32 migration cases. `make legacy-wheel-test` copies their
test and data outside the checkout and runs them alongside the existing 176
compatibility cases, for 208 total. The normal gate needs no network access to the
original package because its checked-in pickles and expected observations are
already recorded. This selected Linux/Python/NumPy matrix does not establish
compatibility with every historical/private build or arbitrary saved object.
