# Experiment source branches

All snapshots are based on `25c8c1642029a640b153c160c22a89795d29ad91`. They preserve the last local prototype/test state, which may contain later negative-control tests beyond the frozen measured patch. Use the original patch/provenance hashes for measured-candidate reconstruction. All twenty original `day/patches/E*/candidate.patch` files are archived; null fields in the old ledger do not mean the patch was absent.

These are research branches, not recommended merges. E12's validated integration is PR #9 (`review/4-numpy-output-copy`), separate from its original prototype.

| Experiment | Snapshot branch | Snapshot commit |
| --- | --- | --- |
| E01 | [archive/20261018/day20-accuracy](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-accuracy) | `142df1280616f1851c25ad0f1d808314822a62d4` |
| E02 | [archive/20261018/day20-speed](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-speed) | `b001268d6559960d83b6959af154a33215c8ad9d` |
| E03 | [archive/20261018/day20-two-run](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-two-run) | `5eef8e0923ae0bb4a8bbe4ce9d5f588c2952753d` |
| E04 | [archive/20261018/day20-scalar-cdf](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-scalar-cdf) | `9809113def8c16a27427b1efc04f806a17726f49` |
| E05 | [archive/20261018/day20-vector-move](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-vector-move) | `e8bad4555da2647059545e27b1343aa52435a144` |
| E06 | [archive/20261018/day20-bucket-tail](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-bucket-tail) | `b46690ac2fbc366ff1ac35cc46be14d72dd19119` |
| E07 | [archive/20261018/day20-scale-hoist](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-scale-hoist) | `5722606d55825179bb3dd0ca48326deaafabc210` |
| E08 | [archive/20261018/day20-heap-root](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-heap-root) | `fd935c2d42f9b40c4f95ce80595377286500ed1c` |
| E09 | [archive/20261018/day20-cap-count](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-cap-count) | `a61fc430d0371e2323127e0d03952ebf5fa7a866` |
| E10 | [archive/20261018/day20-frontend](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-frontend) | `dadd511d9a44fabc1c60e58f4bf918157595076d` |
| E11 | [archive/20261018/day20-numpy-input](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-numpy-input) | `f658994cf5cae45ac03a96d83fabb0adef11c946` |
| E12 | [archive/20261018/day20-numpy-output](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-numpy-output) | `9633ab4834b0a7f17467d033a92fb80e6f5f38e6` |
| E13 | [archive/20261018/day20-python-bytes](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-python-bytes) | `fdf69d14a333245a21ec04cef405f825b8c5e491` |
| E14 | [archive/20261018/day20-polars-decode](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-polars-decode) | `5e0667c3843bb03a7198039eb9dd1b42357b6130` |
| E15 | [archive/20261018/day20-finite-mean](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-finite-mean) | `bfbd5f0ebb78c39dc8dab3ea54ba05acde5df168` |
| E16 | [archive/20261018/day20-subnormal-compress](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-subnormal-compress) | `80aa1105bf55e68762615ff8b1de781ffbcfe4e5` |
| E17 | [archive/20261018/day20-quantile-prefix](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-quantile-prefix) | `40b0f23880c590499ea875780f11be9b345e0a3f` |
| E18 | [archive/20261018/day20-direction](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-direction) | `bf700c6c31312c205cfb53c6a3644d1da03aa0b0` |
| E19 | [archive/20261018/day20-gap-cap](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-gap-cap) | `0a3f4afcda62f641ad0dbaf53b188967b9d62e18` |
| E20 | [archive/20261018/day20-sorted-cdf](https://github.com/ingolfured/gr-tdigest/tree/archive%2F20261018%2Fday20-sorted-cdf) | `e42760fe173257981d8dc0efe9fab1dd5a5ea4f3` |

The `archive/20261018/day20-harness` branch preserves untracked harness/protocol sources. Its only synthetic tracked-file edit is a CHANGELOG note about archiving. The complete branch map also preserves local historical candidate commits and the separate upstream fork's superseded head.
