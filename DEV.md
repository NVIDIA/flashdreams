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
3. Push to `main`.  CI builds the wheel and uploads it.

## What gets published

Only `flashdreams` is published to PyPI (pure-Python wheel, `py3-none-any`).
The `publish-pypi` job in `.github/workflows/ci.yml` uploads to production
PyPI on pushes to `main` after the CPU and GPU jobs pass.

## Integration packages (git-installable)

Integration packages are not published to PyPI.  External consumers
install them from the git repo:

```bash
pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations/wan21"
```

Or with uv:

```bash
uv pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations/wan21"
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
