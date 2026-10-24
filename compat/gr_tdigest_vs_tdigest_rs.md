# G-Research tdigest-rs compatibility

Existing applications retain their original code:

```python
from tdigest_rs import TDigest
digest = TDigest.from_array(values, delta=300)
```

This interface retains the original numerical calculations and public Python
wrapper from G-Research tdigest-rs commit
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. It does not route finite values through
the new compressor. Float32 calculations remain float32, weights remain uint32,
merge/update return new digests, and copy/pickle/dictionary reconstruction keeps
the original default-delta behavior. Cached arrays, native class names, input
conversion, and query handling retain the original interface. Upstream quirks
are intentionally preserved here, including extrapolation at the lower tail.

The new engine is explicitly selected with `from gr_tdigest import TDigest`.
Matching delta does not imply identical numbers between the two engines.
For example, means `[0,10]` and weights `[90,10]` give median `1` in the original
interface and `0` in the modern engine. Modern corrections and state handling
also apply through Polars, Java, and the CLI. They do not silently change the
existing interface. See the [review guide](REVIEW_GUIDE.md) for the numerical
change policy and the remaining release decisions.

## Validation

```bash
make build
make test                 # Full project, original 91 cases, 70 exact-output cases
make legacy-strict-test   # Original suite and numerical source checksums
make legacy-wheel-test    # Installed release Linux wheel: 91 upstream + 70 numerical + 13 API + 2 import cases
```

The original test/support files and benchmark are unchanged; see
[test provenance](tdigest-rs-upstream/PROVENANCE.md). The numerical source has only three module-import adaptations; calculations
and the public Python wrapper remain unchanged; see
[implementation provenance](../src/upstream/PROVENANCE.md).
[Exact-output fixtures](numerical-parity/README.md) exercise both precisions,
compression settings, large integer weights, non-finite input, merge/update,
copy/pickle/dict, centroids, quantiles, medians, and trimmed means. Finite outputs
are checked bit-for-bit, with no relaxed tolerance. NaN payload bits are excluded.

CI runs the full test suite and clean installed release-wheel checks outside the source
checkout. Passing the old tolerance-based suite alone was insufficient: the
previous shared-core adapter passed all 91 tests while changing outputs. The
separate original calculation path addresses that incompatibility. These are
Linux x86_64 regression checks, not proof across every possible platform/input.

Current packaging supports standard CPython3.10–3.15 and NumPy1.26/2 on the
older runtimes, NumPy2 on3.13–3.15. Polars2.0.0 is an optional extra matched
to the plugin ABI; numerical-only installations do not require Polars. The
release gate installs one wheel on every advertised minor and rebuilds the
verified source archive on3.12/3.14/3.15. This does not publish a `tdigest-rs`
release or establish portability across untested operating systems.

## Accuracy comparison

Modern is not uniformly more accurate than this preserved implementation. On the recorded grouped results, Modern wins all four metrics on Bike data; on Energy data, Modern wins average rank, tail rank and average value error, while GR wins peak value error. Several synthetic distributions favor GR. See the [winner and error-ratio tables](quality-20261020/RESULTS.md) and [comparison conclusions](../comparison_design.md#modern-versus-original-gr-measured-accuracy-ci-21-october-2026). All table ratios use Modern error divided by GR error: below 1 favors Modern; above 1 favors GR.

## Benchmark scope

`make legacy-bench-smoke` runs a small custom threaded workload;
`make compat-bench` uses that runner's larger defaults. The copied original
benchmark is retained for reference. Neither target establishes a performance
improvement over the original numerical implementation.

## Test ownership

Keep copied upstream sources/tests unchanged unless deliberately updating the
pin with review. Never regenerate expected numbers merely to make a changed
engine pass. Modern cross-surface coherence remains a separate required gate.
