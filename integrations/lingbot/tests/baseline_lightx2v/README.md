# LingBot LightX2V baseline

Self-contained baseline harness for upstream
[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V), modeled after
`scripts/lingbot/run_lingbot_fast_i2v.sh` with repo-local paths.

## What this harness does

1. Clones `ModelTC/LightX2V` and checks out a pinned commit.
2. Creates an isolated `.venv` under this directory.
3. Installs LightX2V from source (`uv pip install -v .` in the cloned repo).
4. Installs `huggingface_hub[cli]`.
5. Installs local `flashdreams` editable package with `--no-deps` for
   environment alignment.
6. Downloads `robbyant/lingbot-world-base-cam` and
   `robbyant/lingbot-world-fast` if missing.
7. Runs:

```bash
python -m lightx2v.infer \
  --model_cls lingbot_world_fast \
  --task i2v \
  --model_path <local lingbot-world-base-cam path> \
  --config_json <local LightX2V lingbot config> \
  --prompt "<prompt>" \
  --negative_prompt "" \
  --image_path <local lingbot example image> \
  --action_path <local lingbot example action dir> \
  --save_result_path <local output mp4>
```

## Run

```bash
bash integrations/lingbot/tests/baseline_lightx2v/run.sh
```

## Output

- `LightX2V/save_results/output_lightx2v_lingbot_fast_i2v.mp4`
