# Independent preparation review

The corrected-source study is prepared but unexecuted. Full candidate gates and
the assigned CPU/cache slot are prerequisites for its first actual build/smoke.
The historical 52f6212 directory is unchanged. Its selected parameters do not
exist: no training or held-out run occurred there.

The new scientific design equals the historical design exactly except for modern
commit `ff73cfab31d6368d203c1a361b8efaaea29d735b`. The original remains
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Generator, reference math, query grid,
input casting, routes, seeds, delta grid, budgets and selector are unchanged.
The static preparer asserts this equality and checks the four reference-file
hashes. It did not run the rational oracle or generate any library data.

## Selection and interpretation

- Storage uses the largest grid delta whose worst training logical retained
  bytes fit the cap across every declared dataset and supported route. It does
  not choose by error or claim minimum storage/equal RSS.
- Rank policy uses the smallest grid delta whose worst queried training rank
  error is at most the stated threshold across every dataset and supported
  route. This is separate from the storage caps; it is not a median-error or
  value-error tolerance, and it supplies no future-data guarantee.
- Every grid point is inspected independently. No monotonic storage/accuracy
  assumption hides intermediate failures or lower-delta accuracy improvements.
- Training seeds are 1009/10007/65537. Held-out seeds are
  15485863/32452843/49979687. Expected keys reject other seeds, duplicate states,
  missing routes and unequal training delta pairs. All 24 decisions are global
  per core/precision/policy, with no dataset-specific selection.
- Held-out requires an existing selection lock, complete training, exact
  training metadata/raw hashes, and an equal training-only recomputation. It
  preserves the lock before/after execution. It cannot retune based on held-out
  cap or rank misses; those remain explicit observations in the main results.
- Unavailable selections remain unavailable. A delta25 companion exists only
  because the frozen harness accepts two cores; the driver labels it diagnostic
  and does not treat it as qualifying. If both are unavailable, the cell is
  explicitly skipped with a reason rather than counted as tested.
- Modern k-way routes enter its all-route feasibility check. Direct/sequential
  comparisons and the separate modern-kway/original-sequential comparison must
  retain that distinction. Duplicate selected parameter pairs across policies
  are repeated labeled observations, not additional independent evidence.

## Identity and failure guards

The driver checks the exact policy hash and each frozen policy input before every
phase. Its execution protocol includes its own file hash and the independent
reference package identity. It exports pinned Git objects rather than reading a
possibly modified working tree. The complete exported-source manifest, harness
manifest/source, executable and resolved lock are verified against the immutable
build provenance before phases and after execution; each phase retains copies.
Changing or dropping a required source-manifest member is rejected.

The standalone selector checks the listed source hashes but does not independently
reconstruct the full build manifest. The phase driver supplies that complete-set
check and anchors training metadata in its ledger before consuming the selection.
Run the documented driver sequence; do not treat an arbitrary selector output as
proof that a build or prior phase was valid.

Only smoke builds. Training requires complete 128-case/1,408-state smoke; held-out
requires complete 1,152-case/16,128-state training. Existing output directories
cannot be silently overwritten or resumed. Per-case, phase and global budgets
remain binding; stopped runs preserve partial raw files and fail/incomplete
status. Fourteen synthetic tests pass for these guards. They do not establish
Cargo package compatibility, actual reference correctness or a passing smoke.

The independent original's generated Cargo package is named
`tdigest-original-reference`, with explicit harness dependency alias
`tdigest_core`. The modern workspace's public `tdigest_core` keeps its own name.
This avoids duplicate package name/version identities while leaving every
independent original Rust source byte unchanged. The actual build must still
validate resolution. The harness uses a copied modern lock and records its
offline resolved lock; this is distinct from the original committed-lock oracle.

## Future timing review

The separate draft's 18 maximum pairs / 72 single-core processes, fixed policy
reuse, counterbalanced blocks and duplicate-pair grouping are coherent. Its
fresh seed 86028121 is disjoint from both this training and held-out set. At
n=1,000,000 it extrapolates beyond the training sizes (maximum100,000); no new
error-budget guarantee follows. Actual logical bytes, threshold excess and
value/CDF error must accompany timing rows even when a policy misses. Do not
drop misses from the primary report. Modern scalar CDF has no invented original
counterpart, and modern k-way is not the same route as original sequential merge.
No executor, selected setting or timing result has been produced for that draft.

## Intended phase commands

These commands are for later execution in the new protocol directory, after root
releases the CPU/cache slot. Use a fresh study path and the assigned release cache.

```sh
python3 run_storage_budget.py --execute --phase smoke \
  --study /path/to/fresh/storage-study-ff73cfa \
  --repo /path/to/modern-git-repository \
  --upstream /path/to/independent-original-git-repository \
  --cargo-target /path/to/assigned/release-cache

python3 run_storage_budget.py --execute --phase training \
  --study /path/to/fresh/storage-study-ff73cfa

python3 select_storage_budget.py \
  --training /path/to/fresh/storage-study-ff73cfa/training \
  --output /path/to/fresh/storage-study-ff73cfa/selection-lock.json

python3 run_storage_budget.py --execute --phase held-out \
  --study /path/to/fresh/storage-study-ff73cfa \
  --selection-lock /path/to/fresh/storage-study-ff73cfa/selection-lock.json
```

Inspect the completed phase and reference/invariant gates before proceeding to
the next command. Bare `python3 run_storage_budget.py` only emits the protocol.
