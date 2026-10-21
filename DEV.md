# Versioning and Release Preparation

## Version source of truth

The `flashdreams` package version lives in
`flashdreams/flashdreams/_version.py`:

```python
__version__ = "0.2.0"
```

App and integration packages version independently in their own
`pyproject.toml` files and do not track the `flashdreams` version.

## How to bump the version

1. Edit `__version__` in `flashdreams/flashdreams/_version.py`.
2. Commit.
3. Push to `main`. CI builds and validates the `flashdreams` wheel, then stores
   it as a temporary GitHub Actions artifact. CI does not publish the wheel.

## What gets published

Only `flashdreams` is published to PyPI (pure-Python wheel, `py3-none-any`).
Automatic PyPI publication is intentionally disabled while the release process
is updated to scan the final wheel for malware before publication. Do not
publish a wheel built by CI unless the exact artifact has completed the
approved malware-scanning process.

After the CPU tests pass, CI stores the wheel, `SHA256SUMS`, and
`build-provenance.json` for 30 days under an artifact name containing the full
source commit and workflow attempt. This artifact is a temporary handoff for
the controlled malware-scanning release process. Before promotion, the entire
CI workflow, including the required GPU job, must have succeeded for that exact
commit.

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
| flashdreams-causal-forcing | git only | independent |
| flashdreams-cosmos-predict2 | git only | independent |
| flashdreams-fastvideo-causal-wan22 | git only | independent |
| flashdreams-flashvsr | git only | independent |
| flashdreams-lingbot | git only | independent |
| flashdreams-omnidreams | git only | independent |
| flashdreams-self-forcing | git only | independent |
| flashdreams-wan21 | git only | independent |
| flashdreams-wan22 | git only | independent |
| ludus-renderer | git only | independent (0.9.0) |

## PyPI credentials

The current CI workflow does not publish to PyPI and does not consume a PyPI
token. Publication credentials must not be reintroduced into CI without the
approved malware-scanning release gate.
