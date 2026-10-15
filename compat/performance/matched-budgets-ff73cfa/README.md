# Matched-budget timing evidence

Read [RESULTS.md](RESULTS.md) for all 18 paired cells and explicit policy misses. Numeric tables, batches and provenance are in [results](results/analysis-validation.json). [PACKAGE-MANIFEST.json](PACKAGE-MANIFEST.json) hashes the compact bundle.

[PLAN.md](PLAN.md), [PROTOCOL.draft.json](PROTOCOL.draft.json), [CHECKS.json](CHECKS.json) and [preparation-checks.log](preparation-checks.log) are historical preparation records retained unchanged. Their pending/no-execution wording predates the completed build/smoke/full run and is superseded by the result metadata. Frozen hashes must remain unchanged.

Reproduce only with the pinned source/accuracy protocol, completed training selection and independent smoke, following the preserved PLAN and run_timing.py phases. Reanalyze a saved run with `python3 analyze_matched_timing.py RUN_DIRECTORY FRESH_OUTPUT_DIRECTORY`; analysis runs no native code. Full native source exports/data are recorded in [FULL_RESULTS_LOCATION.json](FULL_RESULTS_LOCATION.json), not embedded in this bundle.
