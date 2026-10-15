# Focused interleaved release experiment

This experiment compares the modern core at two immutable commits while measuring
the same pinned original core as a control. It addresses an apparent slowdown in
earlier sequential runs. It is a four-case follow-up, not a general ranking.

- Baseline: `22867bd55739a96a6fab3c8c48b27406ecb32197`.
- Candidate: `c5dec35059990b40fe6f036dc83590bf0cdf2a70`.
- Independent original: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`.

The completed local-date 2026-10-15 run passed eight non-timing smoke processes
and all 32 timed processes. Exact query/centroid/count/sum/wire state matched for
every case and repeat. The earlier double-digit large-case slowdown did not
reproduce consistently across the counterbalanced blocks; original query controls
also shifted. Neither a blanket regression nor a new speedup is established.
Small sorted-case differences remain visible. See the associated results report,
raw JSONL, metadata and analysis JSON for the actual measurements and limits.

## Files and artifact scope

Keep these three portable source files together in this directory:

- `focused_interleaved.py`: immutable exports, release builds, smoke/full schedule,
  hash/state checks and raw output capture.
- `focused_interleaved_tail.rs`: focused build/p50 operations and exact state
  capture. The launcher prepends the unchanged helper code from the historical
  `compat/performance/harness.rs` stored in both commit objects.
- `analyze_interleaved.py`: independently verifies a complete run directory and
  produces descriptive process/block comparisons.

The compact repository evidence consists of the report and captured result JSON
files under `../results/`, with their source IDs and hashes. The source exports,
compiled executables, Cargo cache and complete local run directories are separate
local audit artifacts; this README does not claim they are committed. The strict
analyzer needs the complete run directory, not only the compact evidence files.

## Reproduce

Use Linux/WSL, Python 3.12 or newer, Rust 1.90.0 and cached Cargo dependencies. The
launcher builds offline. Reserve the CPU and any reused Cargo cache against other
build/test activity. No Python extension or existing repository virtualenv is
used for measurements.

Supply a Git repository containing both exact modern commit objects and an
independent original checkout containing its pinned object. Verify availability
with `git cat-file -e COMMIT^{commit}` first. A checkout of a later branch alone
does not guarantee these historical objects are present. Obtain missing historical
objects/source provenance before attempting exact reproduction; substituting HEAD
would be a new experiment and must be labelled accordingly. The launcher makes
fresh source exports and never checks out, resets or edits the supplied repositories.

Without `--execute`, either command prints its plan and performs no build/run:

```sh
python3 focused_interleaved.py --smoke
python3 focused_interleaved.py
```

First run the small API/state gate into a new directory:

```sh
python3 focused_interleaved.py --execute --smoke \
  --repo /path/to/repository-with-both-commits \
  --upstream /path/to/independent/tdigest-rs \
  --output /path/to/new/smoke-run \
  --cargo-target /path/to/reserved/cargo-cache
```

Inspect `metadata.json` for mode `smoke`, status `complete`, eight completed
processes, no source changes and no state mismatches. All four timing fields in
every row must be null. Smoke does not automatically launch the full run.

After that gate passes, use another new output directory:

```sh
python3 focused_interleaved.py --execute \
  --repo /path/to/repository-with-both-commits \
  --upstream /path/to/independent/tdigest-rs \
  --output /path/to/new/full-run \
  --cargo-target /path/to/reserved/cargo-cache
python3 analyze_interleaved.py /path/to/new/full-run
```

Omit `--cargo-target` to create a cache beneath each output directory; a cold cache
can add considerable compilation time. Optional `--cpu` and `--jobs` choose the
logical CPU and build concurrency. Nonempty output directories are rejected.
The recorded full run took 10.7 minutes including its cached release builds;
that is an observation on one machine, not a promised runtime.

## Controls and interpretation

The four cases are 10M uniform and 100K sorted values, each F64/F32, at seed
20261014 and delta 300 (K2Norm, singleton Off). Both cores use the same process
allocator. The resolved dependency locks and modern manifests must match.
Executables are copied to distinct read-only paths and hash-checked before use.

Round one uses ABBA; round two reverses the cases and uses BAAB. Original/modern
operation ordering also reverses between rounds. This gives four independent
process launches per version/case, two in each block. Each operation uses three
warmups, 15 ms calibration and seven batches targeting 25 ms. The inherited timer
sorts batch samples, so their chronological order is unavailable. Do not treat
the seven samples as seven independent experimental repetitions.

Ratios are descriptive medians. Always inspect both blocks and the unchanged
original controls; do not normalize away inconvenient control drift or claim a
speedup from an aggregate that hides opposing block behavior. This run did not
record host temperature, frequency or performance counters. One seed, two blocks,
four cases and one machine cannot establish causality, absence of small overhead,
or a world-wide speed/accuracy ranking. The newer scaling/packaging candidate was
not a measured arm in this historical record.
