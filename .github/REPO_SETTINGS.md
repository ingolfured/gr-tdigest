# GitHub Repository Settings (One-Time)

Use these settings to protect `master` and release tags.

## 1. Protect `master`

GitHub: `Settings` -> `Rules` -> `Rulesets` -> `New branch ruleset`

- Target branches: `master`
- Restrict deletions
- Restrict force pushes
- Require a pull request before merging
- Require status checks to pass before merging:
  - `CI / Lint`
  - `CI / Build and test`

## 2. Protect release tags

GitHub: `Settings` -> `Rules` -> `Rulesets` -> `New tag ruleset`

- Target tags: `v*`
- Restrict creations and updates to maintainers/admins only
- Restrict deletions

## 3. Environments for publishing

GitHub: `Settings` -> `Environments`

Create these environments:
- `pypi`
- `crates-io`
- `maven`

Recommended:
- Add required reviewers for each environment.
- Keep secrets scoped to their matching environment.
