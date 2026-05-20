---
name: release
description: Prepare a release PR for prime-pydantic-config. Use when the user asks to cut, draft, or prepare a new version. Bumps the version in src/pydantic_config/__init__.py and pyproject.toml, drafts release notes under assets/release/, and opens a single PR that the Tag and Release workflow will pick up on merge.
---

# Release

## Goal
Produce one self-contained release PR that the `Tag and Release` workflow can turn into a tag, a PyPI
upload, and a GitHub Release with zero further input.

## When to invoke
The user asks to "cut a release", "release vX.Y.Z", "prepare a release PR", or asks what's needed to
ship a new version.

## Inputs to confirm before acting
1. The target version `X.Y.Z` (semver). If the user did not specify, propose the next patch bump from
   the current `__version__` and confirm before continuing.
2. Scope: are post/dev/pre suffixes wanted? Default to a plain `X.Y.Z` release unless asked.

## Workflow
1. **Read current state**
   - `src/pydantic_config/__init__.py` for the current `__version__`.
   - `pyproject.toml` for the current `version` field — must match `__init__.py`.
   - The most recent prior tag from `git tag --sort=-v:refname | head -1` to anchor the changelog.
2. **Bump the version in two places**
   - Edit `src/pydantic_config/__init__.py`: change `__version__ = "<old>"` to `__version__ = "<new>"`.
   - Edit `pyproject.toml`: change the top-level `version = "<old>"` to `version = "<new>"`. Do not
     touch dependency version specifiers.
3. **Draft release notes**
   - Create `assets/release/RELEASE_v<new>.md` following the template in
     `assets/release/release_workflow.md`.
   - Populate the change list from `git log <prev_tag>..HEAD --oneline` (or `main` history if no prior
     tag), splitting into "Features and enhancements" and "Fixes and maintenance" by commit prefix
     (`feat:`/`fix:`/`chore:` etc.).
   - Set the date to today and include the `Full Changelog` compare link.
4. **Branch, commit, push, open PR**
   - Branch name: `release/v<new>`.
   - Commit message: `release: v<new>`.
   - PR title: `release: v<new>`. PR body should embed the release notes inline so reviewers see them
     without opening the file.
5. **Hand off**
   - Tell the user: "Merge this PR to ship. The `Tag and Release` workflow will tag `v<new>`, publish
     to PyPI via trusted publishing, and create the GitHub Release from the notes file."

## Do not
- Do not push tags yourself. The workflow owns tagging; pushing manually races with it.
- Do not edit `release-dev.yml` or `tag-and-release.yml` while preparing a release.
- Do not include unrelated changes in the release PR. Keep the diff to the three files above.
- Do not skip the release notes file. The workflow fails the auto-tag step if it is missing.

## Sanity checks before opening the PR
- `__version__` and `pyproject.toml`'s `version` are identical strings.
- `assets/release/RELEASE_v<new>.md` exists and renders cleanly.
- No prior tag `v<new>` exists locally or on `origin` (`git ls-remote --tags origin v<new>`).
- The compare link in the notes points from the most recent prior tag to `v<new>`.

## Reference
- Full workflow contract: `assets/release/release_workflow.md`.
- Workflow source: `.github/workflows/tag-and-release.yml`.
