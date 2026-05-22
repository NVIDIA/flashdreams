# OmniDreams Quality Sweep Personal Runbook

This is the end-to-end checklist for syncing the sample data, generating batch
inference videos, running evaluators, and reviewing results in the local web
dashboard.

## 0. Start From The Repo Root

```bash
cd /home/gtong/github/flashdreams
nvidia-smi
```

This machine currently has a single NVIDIA GB300 with enough VRAM for the
Qwen2.5-VL-72B evaluator run, assuming no other large jobs are using the GPU.

## 1. Sync The OmniDreams Data From Hugging Face

```bash
cd /home/gtong/github/flashdreams
bash sync-hf.sh
```

Equivalent explicit command:

```bash
uv run --with "huggingface_hub[cli]" \
  hf download \
  nvidia/omni-dreams-samples \
  --repo-type dataset \
  --local-dir ~/data/omni-dreams-samples
```

Expected input layout after sync:

```text
~/data/omni-dreams-samples/data/single_view/<clip_id>/
  first_frame.png
  prompt.txt
  *_hdmap.mp4
```

Quick check:

```bash
find ~/data/omni-dreams-samples/data/single_view -maxdepth 1 -mindepth 1 -type d | wc -l
find ~/data/omni-dreams-samples/data/single_view -name first_frame.png | wc -l
find ~/data/omni-dreams-samples/data/single_view -name '*_hdmap.mp4' | wc -l
```

## 2. Generate The Batch Input JSON

Smoke manifest:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root ~/data/omni-dreams-samples \
  --output sweep_smoke.json \
  --seeds 0 \
  --limit-items 2
```

Full manifest with three seeds:

```bash
uv run python scripts/generate_omnidreams_sweep_json.py \
  --data-root ~/data/omni-dreams-samples \
  --output sweep.json \
  --seeds 0 1 2
```

Optional quick inspection:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for name in ["sweep_smoke.json", "sweep.json"]:
    path = Path(name)
    if path.exists():
        payload = json.loads(path.read_text())
        print(name, len(payload["items"]), "items")
        print(json.dumps(payload["items"][0], indent=2)[:1200])
PY
```

## 3. Generate Batch-Inferenced Videos

Smoke run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
  --batch-inputs-path sweep_smoke.json \
  --output-dir outputs/omnidreams-quality-sweep-smoke/v1 \
  --batch-results-path outputs/omnidreams-quality-sweep-smoke/v1/manifest.csv
```

Full run:

```bash
uv run --package flash-omnidreams flashdreams-run \
  omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
  --batch-inputs-path sweep.json \
  --output-dir outputs/omnidreams-quality-sweep/v1 \
  --batch-results-path outputs/omnidreams-quality-sweep/v1/manifest.csv
```

The generated rollout directories should contain `video.mp4`. In this workflow,
that `video.mp4` is vertically stacked: HDMap/conditioning on top and generated
RGB video on the bottom.

Quick checks:

```bash
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name video.mp4 | wc -l
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name manifest.csv -print
```

## 4. Run The Scalar Quality Evaluator

The scalar evaluator crops the bottom half of each stacked `video.mp4` into
`video_generated_bottom.mp4`, then scores only that generated crop.

Smoke test:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --limit 1 \
  --max-frames 2 \
  --metrics niqe musiq clipiqa \
  --batch-size 2 \
  --metrics-json-name metrics_smoke.json \
  --summary-path /tmp/omnidreams_metrics_smoke_summary.json \
  --overwrite-metrics
```

Full scalar metrics:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --metrics niqe musiq clipiqa \
  --batch-size 16 \
  --keep-going \
  --overwrite-metrics \
  --summary-path /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep/metrics_summary.json
```

Outputs:

```text
<rollout>/video_generated_bottom.mp4
<rollout>/metrics.json
/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep/metrics_summary.json
```

Interpretation:

- `niqe`: lower is better.
- `musiq`: higher is better.
- `clipiqa`: higher is better.
- These are no-reference quality proxies. They can underrank valid night,
  weather, or unusual scenes, so use them for coarse ranking and seed
  comparison, not final artifact decisions.

## 5. Run The VLM Artifact Evaluator

Use the VLM evaluator for targeted artifacts such as hallucinated vehicles,
scrambled sign glyphs, traffic light issues, lane geometry errors, road-user
anomalies, and temporal inconsistency.

Smoke test without loading a model:

```bash
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

Full VLM artifact run with Qwen2.5-VL-72B:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --backend qwen-local \
  --model-id Qwen/Qwen2.5-VL-72B-Instruct \
  --sample-frames 12 \
  --sheet-columns 3 \
  --thumb-width 640 \
  --overwrite \
  --overwrite-contact-sheets \
  --keep-going
```

Outputs:

```text
<rollout>/vlm_contact_sheet.jpg
<rollout>/vlm_artifacts.json
/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep/vlm_artifacts_summary.json
```

If only the parser/prompt validation logic changed and the existing
`vlm_artifacts.json` files already have `raw_response`, reparse without loading
the model:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_vlm_artifacts.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --reparse-existing \
  --keep-going
```

Interpretation:

- `overall_artifact_severity`: maximum artifact severity across categories.
- Severity `0`: not visible.
- Severity `1`: minor or uncertain artifact.
- Severity `2`: clear artifact that should be reviewed.
- Severity `3`: severe artifact likely to invalidate the clip.
- `needs_review: true`: inspect the clip manually.
- `response_valid: false`: the VLM response schema was malformed or repaired.
  Treat this as needing review even if numeric severities are zero.
- `parse_warnings`: why the response was marked invalid.

## 6. Start The Web Dashboard

Serve from the repo root so the browser can fetch both the dashboard files and
the output videos:

```bash
cd /home/gtong/github/flashdreams
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/
```

The default dashboard loads:

```text
/outputs/omnidreams-quality-sweep/metrics_summary.json
/outputs/omnidreams-quality-sweep/vlm_artifacts_summary.json
```

For another sweep:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?summary=/outputs/another-sweep/metrics_summary.json&vlm=/outputs/another-sweep/vlm_artifacts_summary.json
```

Stop the server:

```bash
fuser -k 8765/tcp
```

## 7. How To Review Results In The Dashboard

Use the dashboard in this order:

1. Start with `Sort: Composite quality` and `View: Best seed per clip` to find
   generally strong outputs.
2. Switch to `Sort: VLM artifact severity` to bring likely artifact failures to
   the top.
3. Use `Artifacts: Needs review` to inspect clips with severity >= 2 or invalid
   VLM schema.
4. Use artifact-specific filters such as `Sign glyph` or `Hallucinated vehicle`
   when looking for a specific failure mode.
5. Use `Video: Generated crop` for quality/artifact inspection. Switch to
   `Stacked source` when you need to compare against the HDMap conditioning.
6. Open the contact sheet from the VLM panel when the evidence mentions frame
   indices or when `response_valid` looks suspicious.

Decision hints:

- A clip with strong scalar metrics and no VLM issues is a good candidate for
  best-quality examples.
- A clip with good scalar metrics but VLM severity 2 or 3 is visually pleasant
  but likely has a semantic artifact.
- A clip with low scalar rank but no VLM issue may simply be a night/rain/low
  contrast scene; inspect before discarding.
- Any schema warning means the VLM answer was not trustworthy. Re-run VLM with a
  stronger model or larger contact sheets before using that result for ranking.

## 8. Useful Summary Commands

Count outputs:

```bash
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name video.mp4 | wc -l
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name metrics.json | wc -l
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name vlm_artifacts.json | wc -l
```

Summarize evaluator status:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep")
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
        print("schema_warnings", sum(1 for r in records if r.get("response_valid") is False))
PY
```

