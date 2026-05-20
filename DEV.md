# Candidates to Support:

- [x] Onmidreams: https://research.nvidia.com/labs/sil/projects/onmidreams/
- [x] Self-Forcing: https://github.com/guandeh17/Self-Forcing
- [x] Causal-Forcing: https://github.com/thu-ml/Causal-Forcing
- [x] FastVideo: https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py

- [] LongCat-Video: http://github.com/meituan-longcat/LongCat-Video/
- [] RollingForcing: https://github.com/TencentARC/RollingForcing
- [] Tencent-HY-WorldPlay: https://github.com/Tencent-Hunyuan/HY-WorldPlay
- [] Matrix-Game: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3
- [] YUME: https://github.com/stdstu12/YUME
- [] LightX2V: https://github.com/ModelTC/LightX2V
- [] OpenWorldLLib: https://github.com/OpenDCAI/OpenWorldLib
- [x] lingbot-world: https://github.com/Robbyant/lingbot-world
- [] Helios: https://github.com/PKU-YuanGroup/Helios
- [] open-oasis: https://github.com/etched-ai/open-oasis
- [] WHAM: https://huggingface.co/microsoft/wham

Servring:
- https://github.com/Rising0321/WMFactory
- https://github.com/softicelee2/aishare/blob/main/271/app.py
- https://github.com/softicelee2/aishare/blob/main/0009/app.py
- https://github.com/inin-zou/Rewind/blob/main/modal-deploy/hy_worldplay_ar.py

---

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

**TEMPORARY:** The CI currently publishes to **test.pypi.org**.  When the
project goes public, switch the CI job to target real PyPI by changing
`--repository testpypi` to `--repository pypi` and the secret from
`TEST_PYPI_API_TOKEN` to `PYPI_API_TOKEN` in `.github/workflows/ci.yml`.

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
| flashdreams | Test PyPI | canonical (from `_version.py`) |
| flashdreams-wan21 | git only | synced |
| flashdreams-self-forcing | git only | synced |
| flashdreams-causal-forcing | git only | synced |
| flashdreams-fastvideo-causal-wan22 | git only | synced |
| flash-onmidreams | git only | synced |
| flashdreams-lingbot | git only | synced |
| ludus-renderer | git only | independent (0.9.0) |

## CI secrets required

| Secret name | Where to create | Purpose |
|-------------|-----------------|---------|
| `TEST_PYPI_API_TOKEN` | https://test.pypi.org/manage/account/token/ | Upload to Test PyPI |
| `PYPI_API_TOKEN` (future) | https://pypi.org/manage/account/token/ | Upload to real PyPI |

Add secrets in GitHub repo Settings -> Secrets and variables -> Actions.
