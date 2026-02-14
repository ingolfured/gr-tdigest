# GitHub Repository Settings

This repository uses GitHub rulesets plus squash-only merges.

## Configuration As Code

- Ruleset spec files:
  - `.github/rulesets/master.json`
  - `.github/rulesets/tags_vstar.json`
- Apply/update script:
  - `scripts/apply_github_rulesets.sh`

GitHub does not automatically apply rulesets from files in the repo.
You still apply them via API (using `gh`), but the desired state is kept in version control.

Apply to current repo:

```bash
./scripts/apply_github_rulesets.sh
```

Apply to a specific repo:

```bash
./scripts/apply_github_rulesets.sh ingolfured/gr-tdigest
```

## Policy (Current)

- `master`:
  - no force-push
  - no deletion
  - direct push allowed
  - PR merges are squash-only (repo merge settings)
- `v*` tags:
  - creation/update/deletion restricted to admin role

## Environments

Create these GitHub environments and scope secrets accordingly:

- `pypi`
- `crates-io`
- `maven`
