# OmniDreams Quality Sweep Runbook

This is the single end-to-end runbook for OmniDreams quality sweeps:

1. Get or prepare input data.
2. Generate a `flashdreams-run` batch manifest.
3. Run batched inference.
4. Run scalar quality metrics.
5. Run the VLM artifact evaluator.
6. Inspect the results in the local dashboard.

The commands below avoid user-specific paths. Set the environment variables for
your machine before running them.

## Path Setup

```bash
export FLASHDREAMS_REPO=/path/to/flashdreams
export OMNI_DREAMS_DATA_ROOT=/path/to/omni-dreams-samples
export NUREC_DATA_ROOT=/path/to/nurec-26.02-release
cd "$FLASHDREAMS_REPO"
```

Check the machine before starting long jobs:

```bash
nvidia-smi
df -h "$FLASHDREAMS_REPO" "$OMNI_DREAMS_DATA_ROOT" "$NUREC_DATA_ROOT"
```

Use Qwen2.5-VL-72B for stronger artifact judgments when enough GPU memory is
available. Use Qwen2.5-VL-7B for faster smoke tests or constrained systems.

## 1. Get Data

Choose one dataset path for the sweep.

### Option A: OmniDreams Sample Data

```bash
uv run --with "huggingface_hub[cli]" \
  hf download \
  nvidia/omni-dreams-samples \
  --repo-type dataset \
  --local-dir "$OMNI_DREAMS_DATA_ROOT"
```

Expected layout:

```text
$OMNI_DREAMS_DATA_ROOT/data/single_view/<clip_id>/
  first_frame.png
  prompt.txt
  *_hdmap.mp4
```

Quick check:

```bash
find "$OMNI_DREAMS_DATA_ROOT/data/single_view" -maxdepth 1 -mindepth 1 -type d | wc -l
find "$OMNI_DREAMS_DATA_ROOT/data/single_view" -name first_frame.png | wc -l
find "$OMNI_DREAMS_DATA_ROOT/data/single_view" -name '*_hdmap.mp4' | wc -l
```

### Option B: NuRec 26.02 Front-Wide Data

Authenticate first if the gated dataset requires it:

```bash
huggingface-cli login
```

Query qualifying front-wide scene-camera items without downloading files:

```bash
uv run python scripts/prepare_nurec_hf_dataset.py \
  --camera camera_front_wide_120fov \
  --output-root "$NUREC_DATA_ROOT" \
  --dry-run
```

Prepare only HDMap videos, prompts, and first frames. The script skips `.usdz`
files. The RGB source video is used to extract `first_frame.png` and is not kept
unless `--keep-rgb-video` is passed.

```bash
uv run python scripts/prepare_nurec_hf_dataset.py \
  --camera camera_front_wide_120fov \
  --output-root "$NUREC_DATA_ROOT"
```

Expected layout:

```text
$NUREC_DATA_ROOT/data/single_view/<scene-id>__camera_front_wide_120fov/
  camera_front_wide_120fov_hdmap.mp4
  first_frame.png
  prompt.txt
  source.json
```

Quick check:

```bash
find "$NUREC_DATA_ROOT/data/single_view" -maxdepth 1 -mindepth 1 -type d | wc -l
find "$NUREC_DATA_ROOT/data/single_view" -name '*_hdmap.mp4' | wc -l
find "$NUREC_DATA_ROOT/data/single_view" -name first_frame.png | wc -l
find "$NUREC_DATA_ROOT/data/single_view" -name prompt.txt | wc -l
```

## 2. Generate The Batch Manifest

The manifest is consumed by `flashdreams-run --batch-inputs-path`. Each item
contains paths to the HDMap video, first frame, prompt, seed, prompt ID, and
optional per-item `total_blocks`.

### OmniDreams Sample Sweep

Smoke manifest:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root "$OMNI_DREAMS_DATA_ROOT" \
  --output sweep_smoke.json \
  --seeds 0 \
  --limit-items 2
```

Full manifest:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root "$OMNI_DREAMS_DATA_ROOT" \
  --output sweep.json \
  --seeds 0 1 2
```

### NuRec Front-Wide Sweep

NuRec HDMap videos are not always aligned to the OmniDreams chunk schedule. Use
`--match-hdmap-duration` to write per-item `total_blocks`, then pass
`--pad-final-hdmap-chunk True` during inference. The runner pads only the final
conditioning chunk and crops the saved stacked video back to the source HDMap
frame count.

Smoke manifest:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root "$NUREC_DATA_ROOT" \
  --dataset nurec-26.02-release \
  --camera-name camera_front_wide_120fov \
  --output nurec_26_02_front_wide_smoke.json \
  --seeds 0 \
  --limit-items 2 \
  --match-hdmap-duration
```

Full manifest:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root "$NUREC_DATA_ROOT" \
  --dataset nurec-26.02-release \
  --camera-name camera_front_wide_120fov \
  --output nurec_26_02_front_wide_sweep.json \
  --seeds 0 \
  --match-hdmap-duration
```

Inspect any manifest:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for name in [
    "sweep_smoke.json",
    "sweep.json",
    "nurec_26_02_front_wide_smoke.json",
    "nurec_26_02_front_wide_sweep.json",
]:
    path = Path(name)
    if not path.exists():
        continue
    payload = json.loads(path.read_text())
    items = payload["items"] if isinstance(payload, dict) else payload
    print(name, len(items), "items")
    if items:
        print(json.dumps(items[0], indent=2)[:1200])
PY
```

## 3. Run Batched Inference

The batch runner loads and tunes the model once for the process, then iterates
over manifest items. This is the preferred path for large sweeps.

Set the recipe name once:

```bash
export OMNIDREAMS_RECIPE=omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf
```

### OmniDreams Sample Sweep

Smoke run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  "$OMNIDREAMS_RECIPE" \
  --batch-inputs-path sweep_smoke.json \
  --output-dir outputs/omnidreams-quality-sweep-smoke/v1 \
  --batch-results-path outputs/omnidreams-quality-sweep-smoke/v1/manifest.csv
```

Full run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  "$OMNIDREAMS_RECIPE" \
  --batch-inputs-path sweep.json \
  --output-dir outputs/omnidreams-quality-sweep/v1 \
  --batch-results-path outputs/omnidreams-quality-sweep/v1/manifest.csv
```

### NuRec Front-Wide Sweep

Smoke run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  "$OMNIDREAMS_RECIPE" \
  --batch-inputs-path nurec_26_02_front_wide_smoke.json \
  --output-dir outputs/nurec-26.02-quality-sweep/tot-main-smoke/v1 \
  --batch-results-path outputs/nurec-26.02-quality-sweep/tot-main-smoke/v1/manifest.csv \
  --pad-final-hdmap-chunk True
```

Full run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  "$OMNIDREAMS_RECIPE" \
  --batch-inputs-path nurec_26_02_front_wide_sweep.json \
  --output-dir outputs/nurec-26.02-quality-sweep/tot-main/v1 \
  --batch-results-path outputs/nurec-26.02-quality-sweep/tot-main/v1/manifest.csv \
  --pad-final-hdmap-chunk True
```

Each rollout directory should contain:

```text
video.mp4
stats.json
meta.json
```

`video.mp4` is vertically stacked: HDMap or conditioning video on top and
generated RGB video on the bottom.

Quick checks:

```bash
export SWEEP_OUTPUT_ROOT="$FLASHDREAMS_REPO/outputs/omnidreams-quality-sweep"
# Or:
# export SWEEP_OUTPUT_ROOT="$FLASHDREAMS_REPO/outputs/nurec-26.02-quality-sweep/tot-main/v1"

find "$SWEEP_OUTPUT_ROOT" -name video.mp4 | wc -l
find "$SWEEP_OUTPUT_ROOT" -name manifest.csv -print
```

Optional NuRec duration check:

```bash
uv run python - <<'PY'
import csv
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["SWEEP_OUTPUT_ROOT"])
manifest = root / "manifest.csv"

def probe(path: str) -> tuple[int, float]:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,duration", "-of", "json", path,
    ], text=True, timeout=10))
    stream = data["streams"][0]
    return int(stream["nb_frames"]), float(stream["duration"])

bad = []
for row in csv.DictReader(manifest.open()):
    meta = json.loads(Path(row["meta_json"]).read_text())
    hdmap = meta["source_record"]["hdmap_path"]
    video = row["output_video"]
    if probe(hdmap) != probe(video):
        bad.append(row["clip_id"])

print("duration_mismatches", len(bad))
print("first_bad", bad[:10])
PY
```

## 4. Run Scalar Quality Metrics

The metrics evaluator crops the bottom half of each stacked `video.mp4` into
`video_generated_bottom.mp4`, then evaluates only that generated crop. If the
input videos are already generated-only, pass `--input-is-generated` to skip the
crop.

Set the output root to evaluate:

```bash
export EVAL_ROOT="$FLASHDREAMS_REPO/outputs/omnidreams-quality-sweep"
# Or:
# export EVAL_ROOT="$FLASHDREAMS_REPO/outputs/nurec-26.02-quality-sweep/tot-main/v1"
```

Smoke test:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py \
  --root "$EVAL_ROOT" \
  --limit 1 \
  --max-frames 2 \
  --metrics niqe musiq clipiqa \
  --batch-size 2 \
  --metrics-json-name metrics_smoke.json \
  --summary-path /tmp/omnidreams_metrics_smoke_summary.json \
  --overwrite-metrics
```

Full metrics:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py \
  --root "$EVAL_ROOT" \
  --metrics niqe musiq clipiqa \
  --batch-size 16 \
  --keep-going \
  --overwrite-metrics \
  --summary-path "$EVAL_ROOT/metrics_summary.json"
```

Outputs:

```text
<rollout>/video_generated_bottom.mp4
<rollout>/metrics.json
$EVAL_ROOT/metrics_summary.json
```

Interpretation:

- `niqe`: lower is better.
- `musiq`: higher is better.
- `clipiqa`: higher is better.
- These are no-reference quality proxies. They can underrank valid night,
  weather, low-contrast, or unusual scenes. Use them for relative ranking and
  triage, not as ground truth.
- Look for agreement across metrics. If metrics disagree, inspect the clip.

## 5. Run The VLM Artifact Evaluator

Run scalar metrics first. The VLM evaluator discovers successful `metrics.json`
files, samples frames from `video_generated_bottom.mp4`, builds a contact sheet,
and asks the configured backend to score targeted artifacts.

Target artifact categories include:

```text
hallucinated_vehicle
sign_glyph
traffic_light
lane_geometry
road_user_anomaly
temporal_inconsistency
```

Smoke test without loading a model:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root "$EVAL_ROOT" \
  --limit 1 \
  --sample-frames 8 \
  --prepare-only \
  --output-name vlm_artifacts_smoke.json \
  --contact-sheet-name vlm_contact_sheet_smoke.jpg \
  --summary-path /tmp/omni_vlm_artifacts_smoke_summary.json \
  --overwrite \
  --overwrite-contact-sheets
```

Full local Qwen run:

```bash
export VLM_MODEL_ID=Qwen/Qwen2.5-VL-72B-Instruct
# Lower-memory fallback:
# export VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root "$EVAL_ROOT" \
  --backend qwen-local \
  --model-id "$VLM_MODEL_ID" \
  --sample-frames 12 \
  --sheet-columns 4 \
  --thumb-width 512 \
  --max-new-tokens 1536 \
  --keep-going \
  --overwrite \
  --overwrite-contact-sheets
```

To preserve a second VLM pass beside an existing one, use alternate output
names:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root "$EVAL_ROOT" \
  --backend qwen-local \
  --model-id Qwen/Qwen2.5-VL-72B-Instruct \
  --sample-frames 12 \
  --sheet-columns 4 \
  --thumb-width 512 \
  --max-new-tokens 1536 \
  --output-name vlm_artifacts_qwen72b.json \
  --contact-sheet-name vlm_contact_sheet_qwen72b.jpg \
  --summary-path "$EVAL_ROOT/vlm_artifacts_qwen72b_summary.json" \
  --keep-going \
  --overwrite \
  --overwrite-contact-sheets
```

If prompt or parsing logic changes and existing `vlm_artifacts.json` files have
`raw_response`, reparse without loading the model:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root "$EVAL_ROOT" \
  --reparse-existing \
  --keep-going
```

Outputs:

```text
<rollout>/vlm_contact_sheet.jpg
<rollout>/vlm_artifacts.json
$EVAL_ROOT/vlm_artifacts_summary.json
```

Interpretation:

- `overall_artifact_severity`: maximum artifact severity across categories.
- Severity `0`: not visible.
- Severity `1`: minor or uncertain artifact.
- Severity `2`: clear artifact that should be reviewed.
- Severity `3`: severe artifact likely to invalidate the clip.
- `needs_review: true`: inspect manually.
- `response_valid: false`: the VLM response schema was malformed or repaired.
  Treat this as needing review even if numeric severities are zero.
- `parse_warnings`: why the response was marked invalid.

The VLM result is a candidate signal, not ground truth. Spot-check clips,
especially when comparing models or when a model flags many review-level
artifacts.

## 6. Inspect Results In The Dashboard

Serve from the repository root so the browser can fetch both dashboard assets
and output videos:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/
```

Useful URLs:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?summary=/outputs/omnidreams-quality-sweep/metrics_summary.json&vlm=/outputs/omnidreams-quality-sweep/vlm_artifacts_summary.json

http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?summary=/outputs/nurec-26.02-quality-sweep/tot-main/v1/metrics_summary.json&vlm=/outputs/nurec-26.02-quality-sweep/tot-main/v1/vlm_artifacts_summary.json
```

If the dashboard has a preset for the run:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?run=nurec-front-wide-qwen72b
```

Review order:

1. Start with `Sort: Composite quality` and `View: Best seed per clip`.
2. Switch to `Sort: VLM artifact severity` for likely artifact failures.
3. Use artifact filters such as `Sign glyph` or `Hallucinated vehicle` for
   targeted review.
4. Use `Video: Generated crop` for quality and artifact inspection.
5. Use `Stacked source` when comparing against the HDMap conditioning.
6. Open the VLM contact sheet when frame evidence or schema validity is
   suspicious.

Decision hints:

- Strong scalar metrics plus no VLM issues: candidate best-quality output.
- Strong scalar metrics plus VLM severity 2 or 3: likely semantic artifact.
- Low scalar rank plus no VLM issue: may be a valid night, rain, or low-contrast
  scene. Inspect before discarding.
- Any VLM schema warning: re-run with a stronger model or inspect manually.

Stop the server:

```bash
fuser -k 8765/tcp
```

## 7. Useful Summary Commands

Count outputs and evaluator files:

```bash
find "$EVAL_ROOT" -name video.mp4 | wc -l
find "$EVAL_ROOT" -name video_generated_bottom.mp4 | wc -l
find "$EVAL_ROOT" -name metrics.json | wc -l
find "$EVAL_ROOT" -name vlm_artifacts.json | wc -l
```

Summarize evaluator status:

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
for name in ["metrics_summary.json", "vlm_artifacts_summary.json"]:
    path = root / name
    if not path.exists():
        print(name, "missing")
        continue
    data = json.loads(path.read_text())
    print(name, data.get("status_counts", data.get("status")))
    if name.startswith("vlm"):
        records = data.get("records", [])
        print("needs_review", sum(1 for r in records if r.get("needs_review")))
        print(
            "schema_warnings",
            sum(1 for r in records if r.get("response_valid") is False),
        )
PY
```

Clean smoke evaluator files if needed:

```bash
find "$EVAL_ROOT" \
  \( -name metrics_smoke.json -o -name vlm_artifacts_smoke.json -o -name vlm_contact_sheet_smoke.jpg \) \
  -delete
```
