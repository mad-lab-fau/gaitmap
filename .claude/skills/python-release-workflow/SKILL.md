---
name: python-release-workflow
description: Use when releasing gaitmap from this repo with uv+poe+GitHub Actions+Read the Docs; supports patch/minor/major and enforces CI+RTD gates before creating a GitHub release.
---

# Python Release Workflow

## Scope
- Repository: `mad-lab-fau/gaitmap`
- Tooling: `uv`, `poe`, `gh`, Read the Docs
- Release types: `patch`, `minor`, `major`
- Package layout: workspace with `gaitmap` and `gaitmap_mad`

## Inputs
- `bump`: `patch` | `minor` | `major`
- `base_ref`: `master`
- RTD project slug: `gaitmap`
- Release notes source: latest release section in `CHANGELOG.md`

## Semver Quick Rule
- `patch`: fixes/docs/tooling/compat, no breaking API
- `minor`: backward-compatible features
- `major`: breaking behavior/API

## Flow
1. **Preflight**
   - `git status -sb` (must be clean or intentionally scoped)
   - `git log -5 --oneline` (for style and release context)

2. **Local verification (fresh)**
   - `uv run poe ci_check`
   - `uv run poe test`
   - `uv build`
   - `uv run poe docs_clean`

3. **Release prep**
   - Finalize release notes in `CHANGELOG.md`
   - Bump version: `uv run poe version --bump=<patch|minor|major>`
   - Verify touched files (typically):
     - `pyproject.toml`
     - `gaitmap/__init__.py`
     - `packages/gaitmap_mad/pyproject.toml`
     - `packages/gaitmap_mad/src/gaitmap_mad/__init__.py`
     - `uv.lock`
     - `CHANGELOG.md`

4. **Commit strategy**
   - Commit code/docs fixes first (if needed)
   - Commit release prep separately (version + changelog)

5. **Push + CI gate**
   - `git push`
   - `REL_SHA=$(git rev-parse HEAD)`
   - `gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5`
   - `gh run watch <run-id>`
   - Required: workflow conclusion is `success` for the release commit SHA

6. **RTD gate (hard)**
   - Wait for RTD build for release SHA to finish successfully
   - Poll API if needed:
     - `curl -s "https://readthedocs.org/api/v3/projects/gaitmap/builds/?limit=10"`
   - Confirm latest relevant build is successful before release creation

7. **Create GitHub release**
   - Tag format: `vX.Y.Z`
   - `gh release create vX.Y.Z --target master --title "vX.Y.Z" --notes "...from changelog..."`

8. **Post-release checks**
   - Watch publish workflow triggered by release (`Upload Python Packages`)
   - Confirm both publish jobs succeed (`gaitmap`, `gaitmap_mad`)
   - Confirm RTD stable/version docs builds are green

## Hard Gates
- Never create GH release before CI green for the release SHA.
- Never create GH release before RTD green for the release SHA.
- Never claim done without command evidence.

## Fast Command Set
```bash
uv run poe ci_check && uv run poe test && uv build && uv run poe docs_clean
uv run poe version --bump=minor
git push
REL_SHA=$(git rev-parse HEAD)
gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5
gh run watch <run-id>
curl -s "https://readthedocs.org/api/v3/projects/gaitmap/builds/?limit=10"
gh release create vX.Y.Z --target master --title "vX.Y.Z" --notes "<changelog section>"
```
