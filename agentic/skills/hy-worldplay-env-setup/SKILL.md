---
name: hy-worldplay-env-setup
description: Provision and run the HY-WorldPlay WAN-5B I2V integration end-to-end — the isolated parity sub-venv (sageattention + accelerate + cloudpickle + torch==2.11 stack), the upstream Tencent-Hunyuan tree clone, the ~52 GB HuggingFace checkpoints (`wan_transformer/` + `wan_distilled_model/model.pt`), the HF_TOKEN requirement, and how to dispatch `flashdreams-run hy-worldplay-wan-i2v-5b` from inside the sub-venv. Use when bootstrapping a new machine for `integrations/hy_worldplay`, when reproducing the parity benchmark, when running the docker convenience wrapper, when extending the plugin's runner config, or when debugging "command not found" / `ImportError: sageattention` / missing-checkpoint errors. Mirrors the layout in `integrations/hy_worldplay/README.md` "Install" + "Staging plan" sections; defer to the README for end-user docs and to this skill for the agent-side decision tree.
---

# HY-WorldPlay environment setup

`integrations/hy_worldplay/` (HY-WorldPlay WAN-5B I2V, phase 1 vendor wrapper) lives in a deliberately **two-venv** layout. The first half of this skill is a decision tree for picking the right venv and provisioning state; the second half is the run path and the bugs you'll see if you skip a step.

## TL;DR

- **Two venvs, on purpose.** The repo-root `uv.lock` carries only the lightweight `flashdreams-hy-worldplay` workspace member (config + runner shim, no heavy deps). The heavy stack — `sageattention`, `accelerate`, `cloudpickle`, `diffusers>=0.34`, `transformers==4.57.6`, `torch==2.11.*` — lives in `integrations/hy_worldplay/tests/parity_check/.venv`, materialized by `uv sync` from that directory.
- **GPU runs go through the parity sub-venv**, regardless of whether you're running the upstream `wan/generate.py` baseline *or* the in-tree `flashdreams-run hy-worldplay-wan-i2v-5b` slug. The two share that sub-venv on purpose so the parity comparison stays apples-to-apples.
- **The CLI surface is `flashdreams-run hy-worldplay-wan-i2v-5b`** (registered via the `flashdreams.runner_configs` entry-point group, same shape as `wan21` / `self_forcing`). There is no `python -m hy_worldplay.cli` — that was removed in PR #103.
- **Three required runtime paths** that have no defaults: `--ar-model-path`, `--ckpt-path`, `--hy-worldplay-repo-root`. Constructing the runner without them raises a clean `ValueError` before any heavy import — designed for fast smoke tests.
- **`HF_TOKEN` is required** to pull `tencent/HY-WorldPlay` model weights from HuggingFace (it's a gated repo).

## 1. Which venv does what

| Layer | Path | What it has | What it can run |
|-------|------|-------------|-----------------|
| Repo-root flashdreams venv | `flashdreams/.venv` (created by `uv sync` at repo root) | `flashdreams-hy-worldplay` workspace member only (config + runner shim). No `sageattention` / `accelerate` / upstream HY tree. | CPU-only smoke tests (`pytest integrations/hy_worldplay/tests/test_smoke.py`). `flashdreams-run --help` lists the slug (entry point is registered) but **invoking it crashes on first heavy import**. |
| Parity sub-venv | `integrations/hy_worldplay/tests/parity_check/.venv` (created by `uv sync` *from that directory*) | Full heavy stack: `sageattention`, `accelerate`, `cloudpickle`, `diffusers>=0.34`, `transformers==4.57.6`, `torch==2.11.*`, plus an editable install of both `flashdreams` and `flashdreams-hy-worldplay`. | Everything: upstream `torchrun wan/generate.py ...` baseline, `flashdreams-run hy-worldplay-wan-i2v-5b ...`, the parity-check script, the docker convenience wrapper. |

If you're not sure which venv `uv` resolves to, `uv` walks upward looking for a `pyproject.toml` — so `cd integrations/hy_worldplay/tests/parity_check && uv run ...` always hits the sub-venv, while `cd /` and use `uv run --project integrations/hy_worldplay/tests/parity_check ...` from anywhere in the repo to target it.

## 2. First-time provisioning checklist

Do these once per machine, in order. Each step is idempotent — rerunning a step that's already done is a no-op (or a fast cache check).

1. **HuggingFace auth.** `tencent/HY-WorldPlay` is gated, so you need a token with read access:

   ```bash
   export HF_TOKEN=hf_...
   export HF_HOME=~/.cache/huggingface  # default; only set if you want a different cache
   ```

   Get a token at <https://huggingface.co/settings/tokens>. Without this, the model download step fails with a 401.

2. **Repo-root sync** (lightweight, ~seconds):

   ```bash
   uv sync   # from the repo root; pulls flashdreams + the hy_worldplay workspace member
   ```

   This is enough to run the CPU-only smoke tests but **not** enough to actually generate video.

3. **Parity sub-venv sync + upstream provisioning** (slow on first run, ~30 min including ~52 GB download):

   ```bash
   bash integrations/hy_worldplay/tests/parity_check/run.sh
   ```

   This script chains four things, all idempotent:

   - `git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay` → `integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/`
   - (Optional) `git apply changes.patch` if a `changes.patch` is present (phase 1 ships without one).
   - `huggingface-cli download tencent/HY-WorldPlay --include "wan_transformer/*" "wan_distilled_model/*"` → `HY-WorldPlay/hf_models/` (~52 GB; the `--include` glob is *required*, positional args after the repo id get treated as exact filenames and silently match zero files).
   - `uv sync` inside `tests/parity_check/` → materializes the heavy sub-venv.
   - As a side-effect, runs `wan/generate.py` once to confirm the upstream baseline works.

4. **Sanity-check the install.** From repo root:

   ```bash
   uv run pytest integrations/hy_worldplay/tests/test_smoke.py -q
   ```

   All tests should pass without GPU. The `test_entry_point_registered` test confirms the `flashdreams.runner_configs` entry point loaded; if it's `SKIPPED` you forgot step 2 (`uv sync` from repo root).

## 3. Running inference

Once provisioned, every GPU invocation goes through the parity sub-venv via `uv run --project ...` (or `cd` into the sub-venv directory). Use `flashdreams-run hy-worldplay-wan-i2v-5b` — the slug is the stable user-facing interface that survives the phase-2 refactor.

### Single-GPU

```bash
PARITY=integrations/hy_worldplay/tests/parity_check
TREE="${PARITY}/HY-WorldPlay"

uv run --project "${PARITY}" flashdreams-run hy-worldplay-wan-i2v-5b \
    --image-path "${TREE}/assets/img/test.png" \
    --ar-model-path "${TREE}/hf_models/wan_transformer" \
    --ckpt-path "${TREE}/hf_models/wan_distilled_model/model.pt" \
    --hy-worldplay-repo-root "${TREE}" \
    --num-chunk 1 --pose 'w-4' \
    --output-dir outputs
```

### Multi-GPU (context-parallelism, up to 8 GPUs)

```bash
uv run --project "${PARITY}" torchrun \
    --nproc_per_node=4 --no-python \
    flashdreams-run hy-worldplay-wan-i2v-5b \
    --image-path "${TREE}/assets/img/test.png" \
    --ar-model-path "${TREE}/hf_models/wan_transformer" \
    --ckpt-path "${TREE}/hf_models/wan_distilled_model/model.pt" \
    --hy-worldplay-repo-root "${TREE}" \
    --num-chunk 4 --pose 'w-16' \
    --output-dir outputs
```

`--no-python` tells `torchrun` to `execvp` the `flashdreams-run` console script directly instead of wrapping it in `python <script>`.

### Pose-string grammar

Mirrors upstream's `wan/generate.py`. Tokens are comma-separated; total latent count must equal `num_chunk * 4` (4 latents per chunk).

| token | action | example |
| --- | --- | --- |
| `w-N` / `s-N` | forward / backward | `w-16` |
| `a-N` / `d-N` | strafe left / right | `d-4` |
| `up-N` / `down-N` | pitch | `up-2` |
| `left-N` / `right-N` | yaw | `right-1` |
| custom JSON | full trajectory from `hyvideo/generate_custom_trajectory.py` | `--pose path/to/traj.json` |

### Docker convenience wrapper

`integrations/hy_worldplay/run-docker.sh` boots the flashdreams container, runs the first-time provisioning if needed, and dispatches the runner. Use it when you don't want to manage host-side `uv` state:

```bash
HF_TOKEN=hf_... ./integrations/hy_worldplay/run-docker.sh
IMAGE_PATH=/path/to/first_frame.jpg PROMPT="..." \
    NUM_GPU=4 POSE='w-32' NUM_CHUNK=8 \
    ./integrations/hy_worldplay/run-docker.sh
```

It bind-mounts the repo's parent at `/workspace` and your `~/.cache/huggingface` so the ~52 GB download is reused across container runs.

## 4. Programmatic access

The runner config is a plain `flashdreams.infra.runner.RunnerConfig` subclass — anything you can do via the CLI you can do via `dataclasses.replace`:

```python
from pathlib import Path
from dataclasses import replace

from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B

cfg = replace(
    RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
    image_path=Path("./assets/img/test.png"),
    ar_model_path=Path("/path/to/models/wan_transformer"),
    ckpt_path=Path("/path/to/models/wan_distilled_model/model.pt"),
    hy_worldplay_repo_root=Path("/path/to/HY-WorldPlay"),
    num_chunk=1,
    pose="w-4",
)
runner = cfg.setup()
runner.run()
```

This must run **inside the parity sub-venv** for the same reason as the CLI: the heavy deps aren't in the main flashdreams venv.

## 5. Common errors and what they mean

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named 'sageattention'` | You ran `flashdreams-run hy-worldplay-wan-i2v-5b` from the main flashdreams venv. Heavy deps live in the parity sub-venv only. | Prefix with `uv run --project integrations/hy_worldplay/tests/parity_check ...`. |
| `command not found: flashdreams-run` | The console script lives in the venv's `bin/`. `uv run` puts it on PATH; bare `flashdreams-run` from a fresh shell doesn't. | Use `uv run --project <parity> flashdreams-run ...`. |
| `flashdreams-run --help` doesn't list `hy-worldplay-wan-i2v-5b` | The `flashdreams.runner_configs` entry point isn't registered (you forgot `uv sync` after pulling, or you're in the wrong venv). | `uv sync` (root) and verify `pytest integrations/hy_worldplay/tests/test_smoke.py::test_entry_point_registered` passes. |
| `FileNotFoundError: HY-WorldPlay tree not found at ...` | `--hy-worldplay-repo-root` points at a missing directory. | Run `tests/parity_check/run.sh` once; it provisions the tree under `HY-WorldPlay/`. |
| `ValueError: Both --ar-model-path and --ckpt-path are required` | You forgot one of the three required paths. | Pass all three (`--ar-model-path`, `--ckpt-path`, `--hy-worldplay-repo-root`). |
| `Fetching 0 files: 0it [00:00]` from `huggingface-cli` | Positional args after the repo id were treated as exact filenames. | Use `--include "wan_transformer/*" "wan_distilled_model/*"`. The setup script already does this — don't reinvent. |
| 401 from HuggingFace | `tencent/HY-WorldPlay` is gated; no token or insufficient permissions. | `export HF_TOKEN=hf_...` with read access; accept the model license on the HF web UI if needed. |
| Parity drift > ~5 / 255 mean \|Δ\| vs upstream | Either `torch` version drift between the parity sub-venv and the flashdreams main venv (current pin: `torch==2.11.*` in both), or the `DEFAULT_PROMPT` / `DEFAULT_NEGATIVE_PROMPT` byte-drifted from upstream. | Re-sync `parity_check/pyproject.toml`'s `torch==2.X.*` pin in lockstep with `flashdreams/uv.lock`. The byte-match is guarded by `test_default_prompt_byte_matches_upstream` — fix any drift there first. |

## 6. Pitfalls when extending the plugin

- **Don't add heavy deps (`sageattention`, `accelerate`, `cloudpickle`, ...) to `integrations/hy_worldplay/pyproject.toml`.** They belong in the parity-check sub-venv. Adding them at the workspace member level leaks them into the repo-root `uv.lock` and bloats every contributor's resolved env.
- **Don't add a new `[project.scripts]` entry.** The slug is dispatched via the `flashdreams.runner_configs` entry-point group; a `[project.scripts]` console script would duplicate the surface and confuse users about which command is canonical.
- **Don't subclass `flashdreams.infra.runner.Runner` for phase 1.** `HyWorldPlayWanI2VRunner` is intentionally a plain class with its own `__init__` because the phase-1 wrapper owns its own distributed setup (deferred to upstream's `WanRunner`) and has no flashdreams `StreamInferencePipeline` for the base `Runner.__init__` to construct. The config sets `pipeline=None`; the base `Runner` is wired to skip pipeline construction in that case, so subclassing it would just re-run setup that the upstream tree also runs.
- **Don't drop the `_ensure_upstream_importable` call.** Upstream's `wan/` package imports siblings (`hyvideo`, `models`, `distributed`, `inference`) by bare name; both `<repo_root>` and `<repo_root>/wan` must be on `sys.path` *before* the `wan.generate` import. Without it, the failure surfaces as a cryptic `ImportError` deep inside upstream rather than the clean `FileNotFoundError` the runner raises today.
- **Don't change `DEFAULT_PROMPT` / `DEFAULT_NEGATIVE_PROMPT`.** Both are byte-pinned to upstream's argparse defaults. UMT5 tokenises trailing punctuation / unicode look-alikes as extra tokens, which shifts the embedding and breaks parity. `test_default_prompt_byte_matches_upstream` will catch the regression in CPU-only pytest before it reaches a GPU run.

## 7. Phase-2 horizon (what changes when the WAN 2.2 5B recipe lands)

When `flashdreams/recipes/wan/` grows a native Wan 2.2 5B `WanInferencePipeline` (phase 2a in the README staging plan), this plugin gets refactored:

- The vendor-wrapper `HyWorldPlayWanI2VRunner` is replaced by a real `Runner` subclass driving a `WanInferencePipeline`.
- `pipeline=None` on the config becomes the actual `WanInferencePipelineConfig` literal; the base `Runner.__init__` then handles setup uniformly.
- The parity sub-venv collapses back into the main flashdreams venv (no more `sageattention` / `cloudpickle`; the recipe shares the wan family's attention dispatcher).
- The `flashdreams-run hy-worldplay-wan-i2v-5b` slug **does not change** — it's the stable user-facing interface across phases, which is the whole point of registering it now via the entry-point group.

The phase boundary is gated on a numeric parity check against the phase-1 baseline so the refactor lands on evidence, not eyeballs.
