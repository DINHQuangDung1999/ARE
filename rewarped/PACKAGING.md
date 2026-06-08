# Packaging

1. Update `VERSION` and `CHANGELOG.md`.

2. Check that package can be built and installed.
```bash
python -m pip install build
python -m build
python -m pip install --upgrade dist/rewarped-<version>-<platform-tag>.whl
python -m rewarped.tests
```

3. Create a release commit and merge into `main`.

4. Tag release with `vX.Y.Z`. Push release commit and tag to GitHub.
```bash
git push
git tag vX.Y.Z
git push public vX.Y.Z
```

5. Check action builds and uploads the package to PyPI.

6. Create a new release on GitHub with a tag and title of `vX.Y.Z`. Use the changelog updates as the description.
