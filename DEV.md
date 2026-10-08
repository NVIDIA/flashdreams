# Versioning and PyPI Publishing

## Version source of truth

The canonical version for the entire monorepo lives in
`flashdreams/flashdreams/_version.py`:

```python
__version__ = "0.1.0"
```

All other package `pyproject.toml` files are kept in sync automatically
by `.github/scripts/sync_version.py`, which runs as a pre-commit hook.

## How to bump the version

1. Edit `__version__` in `flashdreams/flashdreams/_version.py`.
2. Commit.  The pre-commit hook updates all integration `pyproject.toml`
   files to match.
3. Merge the version change through the normal PR process. The resulting
   push to `main` runs CI and builds a wheel; publication requires the
   approval steps below.

## Approve and publish a wheel

1. Review a wheel built from the exact full commit SHA of the `main` push
   and obtain approval for binary distribution. Source-release approval
   alone does not cover publishing a wheel. To build from that checked-out
   commit, run `uv build --wheel --package flashdreams`.
2. In GitHub repository **Settings -> Secrets and variables -> Actions ->
   Variables**, create or update the repository variable
   `FLASHDREAMS_PYPI_APPROVED_SHA` with that full commit SHA. Use the final
   commit on `main`, which can differ from the PR head after merging.
3. Open **Actions -> CI**, select the completed **push to `main`** run for
   that SHA, and choose **Re-run all jobs**. This reruns the original event
   with the same commit and ref, including the previously skipped publish
   job. A CLI equivalent is `gh run rerun RUN_ID --repo NVIDIA/flashdreams`.
   Both CPU and GPU jobs must pass before publication.

Leave the variable unset for source-only releases. An unset or mismatched
value skips publication; each new commit requires its own review and
approval. The publish job rebuilds the wheel from the approved commit.
See GitHub's [workflow rerun instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

## What gets published

Only `flashdreams` is published to PyPI (pure-Python wheel, `py3-none-any`).
The `publish-pypi` job in `.github/workflows/ci.yml` uploads to production
PyPI only for a push to `main`, after the CPU and GPU jobs pass and
`FLASHDREAMS_PYPI_APPROVED_SHA` matches the full source commit SHA.
An already-published version is skipped.

## Integration packages (git-installable)

Integration packages are not published to PyPI.  External consumers
install them from the git repo:

```bash
pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations_v2/wan21"
```

Or with uv:

```bash
uv pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations_v2/wan21"
```

## Package inventory

| Package | Published | Version |
|---------|-----------|---------|
| flashdreams | PyPI | canonical (from `_version.py`) |
| flashdreams-causal-forcing | git only | synced |
| flashdreams-cosmos-predict2 | git only | synced |
| flashdreams-fastvideo-causal-wan22 | git only | synced |
| flashdreams-flashvsr | git only | synced |
| flashdreams-hy-worldplay | git only | synced |
| flashdreams-lingbot | git only | synced |
| flashdreams-omnidreams | git only | synced |
| flashdreams-self-forcing | git only | synced |
| flashdreams-wan21 | git only | synced |
| flashdreams-wan22 | git only | synced |
| ludus-renderer | git only | independent (0.9.0) |

## CI secrets required

| Secret name | Where to create | Purpose |
|-------------|-----------------|---------|
| `PYPI_API_TOKEN` | https://pypi.org/manage/account/token/ | Upload `flashdreams` to PyPI |

Add secrets in GitHub repo Settings -> Secrets and variables -> Actions.
