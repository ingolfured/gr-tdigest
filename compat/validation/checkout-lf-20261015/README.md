# Exact-byte checkout policy

The fork's Windows Python 3.10, 3.11 and 3.12 jobs in [run 38665683258](https://github.com/ingolfured/tdigest-rs/actions/runs/38665683258) built and installed their native extensions, then stopped before pytest in the original-source checksum guard:

```text
RuntimeError: original upstream implementation changed: src/upstream/LICENSE
```

The existing Git attributes did not specify line endings for these byte-hashed files. The Linux controls here reproduce that error using a real Git checkout with `core.autocrlf=true`: the license gains 201 CRLF sequences. With the proposed attributes, the same checkout retains the original LF bytes and the unchanged guard passes.

## Narrow change

Each patch changes only `.gitattributes` and `CHANGELOG.md`. The 13 individual `text eol=lf` entries are derived from the six implementation/wrapper pins in `src/upstream/PROVENANCE.md` and the seven copied Python test/support/benchmark pins in `compat/tdigest-rs-upstream/PROVENANCE.md`. The wrapper path is `bindings/python/tdigest_rs/lib.py`.

The checksum values, checkers, numerical source, build manifests and stored source blobs are unchanged. The three previously documented Rust namespace adaptations remain the only transformations reversed by the original-source checker. No broad text conversion policy is introduced.

Patch bases:

- Personal PR8: `8fff555623ca7a472ac38bc9066dbb81534e424d`.
- Upstream PR38: `bc1f93e15c216b14b532bb005a2a2e2e662a08f8`.

Both source checkouts remained clean before and after this work. `git apply --check` passed for each patch against its base. The patches have not been applied by this control.

## Controls and preserved evidence

For each base, `prepare_and_verify.py` exports only the required tracked blobs into a disposable Git index and runs the actual `test_legacy_wheel.verify_upstream_snapshot` function:

1. Original LF bytes: all six implementation/wrapper and seven snapshot checks pass.
2. Existing attributes with `core.autocrlf=true`: the actual guard fails on `src/upstream/LICENSE`, matching remote CI.
3. Proposed attributes with `core.autocrlf=true`: all 13 checks pass; each pinned file equals its source Git blob byte for byte.
4. An added line in the corrected license: the unchanged guard rejects the deliberate mutation.
5. Restored license: all 13 checks pass again.

Unpinned LF and CRLF controls have identical checkout bytes before and after the fix. The private control also includes the actual tracked `compat/accuracy/matched-budgets-ff73cfa/results/policy-outcomes.csv`; its CRLF bytes are unchanged and equal its source Git blob. The source repository tree listings, exact guard/provenance/blob hashes, checkout byte inventories, subprocess exits and log hashes are recorded in `RESULTS.json`.

The three raw remote failure logs are copied without normalization under `ci-failure-logs/`; their SHA256 values and job URLs are in `RESULTS.json`. The controls run on Linux only and do not compile or execute any Windows artifact. They establish the checkout failure mechanism and exact-byte correction; successful native Windows validation still requires the remote CI rerun after the patch is pushed.

The recorded driver contains absolute source paths and base assertions for this session. Reproduction requires those exact clean repositories and the preserved raw failure logs; use a fresh `--output` directory. Existing control directories are never overwritten. `MANIFEST.json` binds the portable files; scratch Git objects and materialized checkout trees are intentionally excluded from that compact bundle.
