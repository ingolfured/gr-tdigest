# Separated-mode accuracy diagnosis

This is a read-only diagnosis of the held-out survey on modern source
`52f6212008111b62fa07374278b850868b9ea77e`, not a new acceptance gate or a
production patch. All probes used the preserved installed Linux wheel whose
native SHA-256 is `78a7473df6e7b20c703a57559b8ea9975848a3f55f745a7fcd1e721a598ece09`.
No build, benchmark timing, or production edit was performed.

## Finding

The large errors are genuine limitations of compression plus interpolation
across well-separated narrow modes. They are not lost mass, incorrect weights,
out-of-order means, or an arithmetic regression demonstrated by these cases.
The Type-7 value difference alone is not a sufficient accuracy oracle: a point
inside a large empty gap can have small rank error. Here the independently
measured CDF and rank errors also show an accuracy limitation.

For seed 999983, n=100000, f32, direct delta=300, the recorded worst probe is:

- x = 9.970972061157227
- modern CDF = 0.5525535257868059
- midpoint empirical CDF = 0.49999
- absolute error = 0.052563525786805954

The critical Mixed centroid has mean 9.591711044311523, weight 3742, and rank
interval [0.49818, 0.53560]. It combines 73 negative-mode samples and 3669
positive-mode samples. Its center rank is 0.51689. The next centroid has mean
9.987577438354492, weight 3703, center rank 0.554115. Interpolating between
these two centers at x=9.970972 gives exactly the recorded 0.5525535257868059.
The true positive mode has barely begun at that x. Information about the gap
was already lost when the mixed centroid was formed; a CDF-only change cannot
recover it reliably from the stored mean and weight.

At n=1000, f64, original and modern medians are exactly
8.814346314338946, versus Type-7 9.982112706476084 and rank error 0.017.
At n=100000, f64, original median is -0.5661056623436598 and modern is
-0.5661056623436691: the same gap behavior with about 9.3e-15 numerical drift,
not bit identity. The original API path retains the original arithmetic.

## Small examples

At the surveyed delta=300, randomness is unnecessary:

```python
import numpy as np
from gr_tdigest import TDigest

values = np.r_[np.linspace(-10.01, -9.99, 483),
               np.linspace(9.99, 10.01, 517)]
d = TDigest.from_array(values, delta=300)
assert len(d) == 128
print(d.quantile(.5))       # 8.82159916955987
print(np.quantile(values,.5))  # 9.990639534883721
print(d.cdf([9.99]))        # [0.5164819569002091]
```

The midpoint empirical CDF at 9.99 is 0.4835. The p50 rank error is 0.017.
This reproduction also works with f32 means (tiny numerical differences).

A five-point example makes the mechanism explicit at a deliberately low
delta, not the surveyed/default delta:

```python
d = TDigest.from_array([-10.01, -9.99, 9.99, 10.0, 10.01], delta=10)
print(d.means)              # [-10.01, 0.0, 10.0, 10.01]
print(d.weights)            # [1.0, 2.0, 1.0, 1.0]
print(d.quantile(.5))       # 5.0; exact median is 9.99
```

The cluster joining -9.99 and 9.99 has mean zero. The query subsequently
interpolates from that Mixed centroid toward the next Atomic point. This
example demonstrates the mechanism; it is not evidence that all five-point
inputs at delta=300 fail, nor that the old query kernel has identical Atomic
interpolation conventions.

## Why f32 changes the particular bad centroid

`compressor.rs` normalizes/coalesces equal means before `klimit_merge`.
The K2Norm implementation passes the resulting item count as n to
`delta / (4*ln(n/delta)+24)`. The original path retains its own original
algorithm. In these direct modern builds:

| Sample size | Mean type | Unique input means / scale n | K2Norm factor | Output centroids |
| --- | --- | --- | --- | --- |
| 1000 | f64 | 1000 | 10.4109221449 | 128 |
| 1000 | f32 | 995 | 10.4181711338 | 128 |
| 100000 | f64 | 100000 | 6.3510112513 | 142 |
| 100000 | f32 | 56987 | 6.6685658885 | 149 |

Coalescing changes both the scale factor and indivisible input piles, moving
cluster boundaries relative to the empty gap. More centroids in the f32 case
did not prevent a worse error at this particular probe. This is a phase and
distribution sensitivity, not evidence that f32 is always less accurate.
No change to the n definition is proposed without separate experiments.

Delta is not a hard centroid or byte budget. Logical retained bytes from the
survey for n=100000 were modern f32 2480 (149 centroids) versus original f32
1184 (142 centroids); modern f64 3512 (142) versus original f64 1752 (142).
The modern layout stores f64 weights plus kind and preserves more state. An
accuracy comparison at equal delta therefore does not imply equal memory.
These are logical retained bytes, not RSS or allocator measurements.

## Bounded follow-up options

1. Keep the compatibility API untouched and disclose this limit. Do not claim
   world-leading accuracy from the current gates: invariants are different
   from approximation quality.
2. Compare capacity/accuracy/time on a fresh, separately seeded evaluation
   matrix. On the deterministic 1000-point reduction, delta 300/600/1200 gave
   128/248/478 centroids; p50 rank error was 0.017/0/0. This is diagnostic
   capacity sensitivity, not a tuned held-out success or universal guarantee.
3. Only in an independent experiment, consider a compression rule that avoids
   spending one cluster across a substantial empty gap, with a defined memory
   budget and merge behavior. Any adaptive rule needs broad distribution and
   merge-route evidence. A blanket gap threshold or epsilon is not justified.
4. Per-centroid support metadata may improve some queries but costs storage,
   wire migration and merge complexity. Bounds alone still do not recover a
   bimodal distribution already merged inside one cluster. Do not introduce
   it as a guaranteed fix based on this example.

Production source locations at the tested revision: `compressor.rs` normalization
at 118-119, weighted mean/kind at 50-85, k-limit at 315-344; `scale.rs` K2Norm
at 68-72; `cdf.rs` center interpolation at 196-232; `quantile.rs` Mixed
interpolation. The compressor is behaving according to those rules here.

## Evidence and reproduction

`comparison.json` preserves all four source-case filenames, raw SHA-256,
post-cast input FNV fingerprints, original/modern medians, centroid counts and
logical bytes. Each `{1000,100000}-{float32,float64}.json` contains every
modern centroid, prefix rank, and its inferred original sample interval.
The latter is valid for this sorted direct build, not an arbitrary merged
digest reconstruction. `reduced.json` records the deterministic probes.

`inspect_separated_modes.py` reimplements the frozen SplitMix64/Box-Muller
generator and asserts all four input fingerprints, median bits and worst-CDF
values match the existing survey before saving evidence. It needs NumPy,
the preserved installed wheel and the raw survey files at the recorded paths.
`reduce_separated_modes.py` needs only the installed wheel and NumPy. Run either
with the preserved environment:

```
/var/tmp/codex-tdigest-pr3-20261014/overnight/package-hygiene/standard-sdist-wheel-52f6212/venv/bin/python inspect_separated_modes.py
```

The short native probes completed before root resumed sole CPU ownership.
No native probe process remains. This bundle is scratch evidence, not a
passing numerical accuracy gate.

## Compact review bundle

This copy omits the complete raw survey and all-centroid dumps. Their hashes
and original paths remain in `comparison.json`; `critical-centroids.json`
contains the causal brackets. `reduced-cases.json` retains the surveyed-delta
example, its f32 counterpart, the five-point mechanism, and capacity controls.

For a portable reproduction with a wheel built from the recorded source:

```sh
python reproduce_reduced.py --output /tmp/separated-reproduction.json
```

This script compares the saved outputs exactly and records native/NumPy
provenance. Success means the known limitation was reproduced, never that
accuracy passed. A different build may fail exact comparison; inspect that
difference rather than silently changing the reference. The portable script
was syntax-checked during packaging, not run while the parent owned CPU.
`recorded/path-specific-runners/` are the original scripts actually executed;
their historical absolute paths intentionally remain unchanged.
