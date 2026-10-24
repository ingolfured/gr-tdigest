# Measured accuracy: 21 October 2026

Read the [accuracy report](../../tests/quality/ACCURACY.md) for the default-setting
gains, comparable-storage scores, cost and query-level explanations. These are
measurements of the exact installed development wheel from runtime source
`aaf7ab3956fed4269c1eaeef915a6faee78fa305`, with unchanged original arithmetic.

- `summary.json`: aggregates, win/tie/loss counts and evidence identities.
- `defaults.json.gz`: all 396 default-compression profiles. Both engines receive
  the same data/precision; original delta300 and modern capacity1000 are actual
  omitted-argument compression defaults. No size matching or accuracy tuning.
- `matched-storage.json.gz`: the 234 eligible K2/1000 established-data rows from
  the complete measured 4,140-profile diagnostic. The parent result hash and
  selection are retained; no eligible loss is removed.
- `query-explanations.json.gz`: fresh reconstruction of all 204 default profiles
  with a loss beyond the existing comparison resolution. Query values, the
  exact sample reference, position errors and numerical value errors stay visible.
- `explain.py`: reproduces that query-level reconstruction from a comparison.
- `plot.py`: draws the committed SVG from the summary using matplotlib.

The [default runner](../../scripts/compare_default_accuracy.py) uses the existing
exact reference and pinned public fixtures. It downloads nothing and refuses to
overwrite evidence. Its exit status reports measurement failure, not universal
per-metric dominance. The recorded raw data/configuration/scripts/native hashes
bind the measurements; `summary.json` records compressed and decoded hashes.

Stored profiles include shared/reversed data and are not independent trials.
Logical field sizes do not measure retained RAM or RSS. These score comparisons
are separate from exact compatibility, state integrity and frozen regressions.
No approximation tolerance, fixture, algorithm or original source is changed.
