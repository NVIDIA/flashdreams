# OmniDreams VLM Artifact Evaluation

This evaluator uses a vision-language model to score targeted scene artifacts
that generic image-quality metrics miss. It is designed for issues such as
hallucinated vehicles, scrambled sign glyphs, malformed traffic lights, lane
geometry errors, and object pop-in across sampled frames.

The current implementation supports a local Qwen2.5-VL backend through Hugging
Face Transformers. The backend boundary is intentionally narrow so a hosted
OpenAI backend can be added later without changing the sweep JSON format.

## Inputs

Run the scalar metrics evaluator first so each rollout has:

```text
video_generated_bottom.mp4
metrics.json
```

The VLM evaluator reads every successful `metrics.json`, samples frames from the
cropped generated video, and writes:

```text
vlm_contact_sheet.jpg
vlm_artifacts.json
```

It also writes a sweep-level summary:

```text
vlm_artifacts_summary.json
```

## Smoke Test Without Loading A Model

Use `--prepare-only` to verify discovery, video decoding, contact-sheet
creation, and JSON writing:

```bash
cd /home/gtong/github/flashdreams
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --limit 1 \
  --sample-frames 8 \
  --prepare-only \
  --output-name vlm_artifacts_smoke.json \
  --contact-sheet-name vlm_contact_sheet_smoke.jpg \
  --summary-path /tmp/omni_vlm_artifacts_smoke_summary.json \
  --overwrite \
  --overwrite-contact-sheets
```

Remove smoke artifacts after inspection:

```bash
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  -name 'vlm_artifacts_smoke.json' -o -name 'vlm_contact_sheet_smoke.jpg' \
  | xargs -r rm -f
```

## Run Local Qwen2.5-VL

The default model is `Qwen/Qwen2.5-VL-7B-Instruct`.
The first real run downloads the model weights if they are not already present
in the local Hugging Face cache.

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --backend qwen-local \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --sample-frames 12 \
  --sheet-columns 4 \
  --thumb-width 512 \
  --keep-going \
  --overwrite \
  --overwrite-contact-sheets
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--limit N` | Evaluate only the first `N` rollouts. |
| `--sample-frames N` | Number of evenly spaced frames to place in the contact sheet. |
| `--thumb-width N` | Width of each sampled frame in the contact sheet. |
| `--dtype auto|bfloat16|float16|float32` | Model dtype. `auto` uses BF16 on CUDA and FP32 on CPU. |
| `--device-map auto` | Let Accelerate place the local model. |
| `--disable-schema-repair` | Disable repair of known malformed Qwen JSON keys. By default repaired responses are kept but marked invalid. |
| `--reparse-existing` | Re-parse saved `raw_response` fields and regenerate `vlm_artifacts.json` / summary without loading the model. |
| `--overwrite` | Recompute existing `vlm_artifacts.json` files. |
| `--overwrite-contact-sheets` | Recreate existing contact sheets. |
| `--prepare-only` | Build contact sheets and placeholder JSON without loading a VLM. |

## Output Schema

Each rollout-level `vlm_artifacts.json` contains:

```json
{
  "status": "ok",
  "backend": "qwen-local",
  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
  "cropped_video": ".../video_generated_bottom.mp4",
  "contact_sheet": ".../vlm_contact_sheet.jpg",
  "sampled_frame_indices": [0, 43, 87],
  "artifacts": {
    "overall_artifact_severity": 2,
    "response_valid": true,
    "parse_warnings": [],
    "needs_review": true,
    "highest_severity_categories": ["sign_glyph"],
    "artifact_scores": {
      "hallucinated_vehicle": {
        "severity": 0,
        "confidence": 0.2,
        "evidence": ""
      },
      "sign_glyph": {
        "severity": 2,
        "confidence": 0.8,
        "evidence": "A roadside sign has scrambled text in frames 87 and 130."
      }
    }
  },
  "raw_response": "{...}"
}
```

Severity rubric:

| Severity | Meaning |
| --- | --- |
| `0` | Not visible |
| `1` | Minor or uncertain artifact |
| `2` | Clear artifact that should be reviewed |
| `3` | Severe artifact likely to invalidate the clip |

Sort the summary by `overall_artifact_severity` or filter
`needs_review == true` to find the most suspicious outputs.

## Response Validation

The evaluator treats the model response schema as part of the result quality.
If Qwen emits common malformed keys such as `artifact_scoresrs` or
`schema_versionion`, the evaluator repairs the keys so the scores are not lost,
but writes `response_valid: false`, adds `parse_warnings`, and marks
`needs_review: true`. If the response cannot be repaired into an
`artifact_scores` object, that rollout is written with `status: failed` and the
raw response is preserved for debugging.

This prevents a malformed all-zero response from being silently interpreted as
a clean clip. Re-run with `--overwrite` after changing prompts or validation
logic so stale `vlm_artifacts.json` files do not remain in the summary.

To update an existing run that already has `raw_response` values without
spending another model pass:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --reparse-existing \
  --keep-going
```
