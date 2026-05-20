# Release workflow

The `Tag and Release` GitHub Actions workflow (`.github/workflows/tag-and-release.yml`) publishes a new
`prime-pydantic-config` version automatically when a release-prep PR lands on `main`. A single PR is the
only input the workflow needs.

## What the workflow does

On every push to `main`, the workflow:

1. Reads `__version__` from `src/pydantic_config/__init__.py` at the previous commit and at HEAD.
2. If the value changed, it cross-checks that `pyproject.toml`'s `version` matches and that
   `assets/release/RELEASE_v<new>.md` exists.
3. Creates an annotated tag `v<new>` and pushes it to `origin`.
4. Builds the sdist and wheel with `uv build`.
5. Publishes to PyPI via OIDC trusted publishing (no token needed; the `pypi` GitHub environment grants
   `id-token: write`).
6. Creates a GitHub Release whose body is `assets/release/RELEASE_v<new>.md` and attaches the built
   `dist/*` artifacts.

The tag push triggered by `GITHUB_TOKEN` does not re-trigger this workflow, so each version runs exactly
once.

## Cutting a release

Open one PR titled `release: vX.Y.Z` that contains exactly:

1. Bump `__version__` in `src/pydantic_config/__init__.py` to `X.Y.Z`.
2. Bump `version` in `pyproject.toml` to the same string.
3. Add `assets/release/RELEASE_vX.Y.Z.md` summarizing the changes (template below).

Verify CI is green, then merge. The workflow handles the rest.

### Release notes template

```markdown
# prime-pydantic-config vX.Y.Z Release Notes

*Date:* MM/DD/YYYY

## Highlights

- <one or two sentences on the most important change>

## Changes since vA.B.C

### Features and enhancements

- feat: ... (#PR)

### Fixes and maintenance

- fix: ... (#PR)

**Full Changelog**: https://github.com/PrimeIntellect-ai/pydantic-config/compare/vA.B.C...vX.Y.Z
```

## Manual re-run

If the workflow needs to be re-run for an existing tag (for example, the PyPI upload failed and was
cleaned up), use **Actions → Tag and Release → Run workflow** and supply the existing tag (e.g.
`v0.4.0`). The job checks out that tag and repeats the build, publish, and release-creation steps.

## Troubleshooting

- **Version mismatch error**: `pyproject.toml` and `src/pydantic_config/__init__.py` must agree. Land a
  fix-up PR that aligns both, then proceed.
- **Missing release notes**: `assets/release/RELEASE_v<version>.md` must exist before the tag is cut.
  Add the file and re-merge.
- **PyPI publish failed**: delete any partially uploaded files from PyPI, then re-run the workflow
  manually with the same tag. PyPI rejects duplicate uploads.
