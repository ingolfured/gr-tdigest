# NumPy output conversion

The modern Python binding now copies its existing f64 result slice directly into NumPy-owned memory. Previously it constructed a Python list and a Python float per value before NumPy copied the result. Values, Float64 dtype, shape, writable/resizable ownership and independent snapshots are preserved. The original `tdigest_rs` API and numerical implementation are unchanged.

The frozen baseline is `25c8c1642029a640b153c160c22a89795d29ad91`. `results.json` records the measured production hash, full heldout initial/repeat classifications and paired ratios. `protocol.json`, `run.py.txt` and `probe.py.txt` are the unchanged measurement protocol and workers; their original paths describe the isolated validation environment, not a portable one-command benchmark. No timing was repeated while integrating this change.

On NumPy 2.3.4 in single-thread WSL Linux, the six qualified nonempty CDF cases used **44.86–54.76% less time** and the six quantile cases used **36.53–51.39% less time**. These use 64, 1,000 or 32,768 queries, f32/f64 digest modes and fresh 8,192-input ties/separated fixtures. Only cells passing all four paired ratios, the 50 ms minimum batch and 10% drift gates contribute to these ranges. These are Python output-path results, not a new Rust numerical or accuracy claim.

All 36 fresh heldout cells were considered: 25 qualified improvements, seven neutral/noisy, two unstable and two still short. No confirmed regression occurred. The f64 encode control was stable/neutral in the original heldout run but unstable in the longer repeat (one ratio 1.1269; candidate drift 1.1285); that repeat remains inconclusive. Empty f32 CDF and two large weights-getter cases also remain inconclusive. The unresolved screen observations are retained, and large apparent short-batch gains are not presented as qualified wins.

Installed-wheel exactness was checked on NumPy 1 and 2. NumPy 1 has no performance claim. Equal returned dtype/shape/ownership establishes equal array payload; allocator usable bytes and peak RSS were not measured. Final integration/release validation is recorded separately.
