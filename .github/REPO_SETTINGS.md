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
  - PR rule active for everyone else
  - admin role can bypass PR requirement and push directly
  - PR merge method restricted to squash
- `v*` tags:
  - creation/update/deletion restricted to admin role

## Environments

Create these GitHub environments and scope secrets accordingly:

- `pypi`
- `crates-io`
- `maven`
