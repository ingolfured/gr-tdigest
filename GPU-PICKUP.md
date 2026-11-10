# Picking up GPU support

Parked on 2026-11-10 in `ingolfured/gr-tdigest`. The branch to use is **`archive/gpu-support`**. It contains all 20 implementation commits from PRs #99–#105. The last code commit is `86f8b73c84ff419048c035df1a0519aef18ffc0c`. The original branches are no longer needed to reconstruct the stack.

## Resume from a fresh clone

```sh
git clone https://github.com/ingolfured/gr-tdigest.git
cd gr-tdigest
git switch --track origin/archive/gpu-support
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

The stack originally branched from `0eaa794`. At parking time, `master` was one commit ahead of that base ([#97](https://github.com/ingolfured/gr-tdigest/pull/97), CDF optimizations). A trial merge reported a conflict only in `CHANGELOG.md`. When updating from the then-current `master`, preserve both sets of entries and rerun the checks below. To put the entire change back up for review, reopen the closed PR titled **Linux HIP GPU support** against `master`. To start a fresh PR from the saved work instead:

```sh
gh pr create --repo ingolfured/gr-tdigest --base master --head archive/gpu-support --draft --title "Linux HIP GPU support" --body "See GPU-PICKUP.md on this branch for scope, prior validation, and remaining work."
```

## Where to start reading

- [#99](https://github.com/ingolfured/gr-tdigest/pull/99) and [#100](https://github.com/ingolfured/gr-tdigest/pull/100) introduce opt-in Linux HIP builds and resident snapshots for batched CDF, quantile, and median queries. Python can chain queries through `DeviceArray`.
- [#101](https://github.com/ingolfured/gr-tdigest/pull/101) and [#102](https://github.com/ingolfured/gr-tdigest/pull/102) implement stable sorting, tie coalescing, compression, construction, and weighted or unweighted updates from resident observations.
- [#103](https://github.com/ingolfured/gr-tdigest/pull/103) implements ordered merges of resident digests with canonical TDIG output. [#104](https://github.com/ingolfured/gr-tdigest/pull/104) exposes explicit GPU operations through Java/JNI and Polars.
- [#105](https://github.com/ingolfured/gr-tdigest/pull/105) adds CPU and GPU `merge_groups`. Independent groups can run concurrently. Within each group, output must match the CPU input-order left fold byte-for-byte. The final commit sizes GPU chunks using kernel occupancy and available memory and retries workspace allocation with a smaller budget.

## First validation pass

The [GPU test guide](tests/gpu/README.md) lists toolkit prerequisites and device settings. From the repository root, run the CPU checks and then the hardware suite. This example uses the RX 7900 XTX:

```sh
make build && make test
HIP_PLATFORM=amd GPU_ARCH=gfx1100 GR_TDIGEST_TEST_DEVICE=0 tests/gpu/check.sh
```

On a two-card machine, set `GR_TDIGEST_SECOND_DEVICE=1` for cross-device coverage, then swap the ordinals to exercise the other card. The suite covers Python, Polars, native Rust, the compressor, and Java/JNI.

At the saved code tip, #105 reports that both RX 7900 XTX suites passed, including the adaptive workspace boundary test. All seven [CPU/package CI checks](https://github.com/ingolfured/gr-tdigest/actions/runs/42588951610) passed. The NVIDIA sm_89 wheel compiled and imported with CUDA 12.8.1. **Running the kernels on NVIDIA hardware is the outstanding validation step.** On a matching card, set `HIP_PLATFORM=nvidia`, `CUDA_PATH`, and the appropriate `GPU_ARCH`, then run the same hardware suite and record parity and timings.

For performance work, start with `python tests/gpu/bench_groups.py --device 0` using the installed HIP wheel, then repeat with `--weighted`. #105 records an advantage for the large integral-weight grouped workload on Radeon and a slower result than grouped CPU for fractional weights. Single-digest construction and ordered single-group merges were also slower on Radeon. Use the workload dimensions and revision-specific timings in #102, #103, and #105 when comparing changes.
