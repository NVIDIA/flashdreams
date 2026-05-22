# OmniDreams Metrics Evaluation

This guide covers how to run no-reference quality metrics on OmniDreams sweep
outputs and how to read the resulting `metrics.json` files.

## What The Evaluator Scores

OmniDreams sweep outputs store each `video.mp4` as a vertical stack:

- Top half: conditioning / HDMap visualization
- Bottom half: generated camera video

The sweep evaluator crops the bottom half first, saves it beside the original as
`video_generated_bottom.mp4`, and evaluates only that cropped copy. This keeps
metrics focused on the generated RGB video instead of scoring the HDMap overlay.

The default metrics are no-reference image-quality metrics computed over video
frames:

| Metric | Better | Typical Meaning |
| --- | --- | --- |
| `niqe` | Lower | Naturalness / distortion estimate. Lower usually means fewer statistical artifacts. |
| `musiq` | Higher | Learned perceptual image quality score, roughly on a 0-100 scale. |
| `clipiqa` | Higher | CLIP-based image quality score, usually 0-1. Often useful for generated imagery. |

These metrics do not compare against ground truth. Use them to rank outputs
within the same sweep, crop, resolution, frame count, and metric configuration.
They are signals, not a replacement for visual inspection.

## Environment

From the repo root:

```bash
cd /home/gtong/github/flashdreams
uv lock --check
```

Run commands with `--extra eval` so optional metric dependencies such as `pyiqa`,
`torchmetrics`, and `tabulate` are available:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py --help
```

The first metric run may download model weights into `~/.cache/torch/hub/pyiqa`.

## Smoke Test

Before a full sweep, evaluate a small subset:

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

This checks video discovery, bottom-half cropping, metric model loading, and JSON
writing without spending time on every frame of every output.

Remove smoke files before publishing final results:

```bash
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  -name 'metrics_smoke.json' -delete
```

## Full Sweep

Run all default metrics on every `video.mp4` under the sweep root:

```bash
uv run --extra eval python scripts/evaluate_omnidreams_sweep_metrics.py \
  --root /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  --metrics niqe musiq clipiqa \
  --batch-size 16 \
  --keep-going \
  --overwrite-metrics \
  --summary-path /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep/metrics_summary.json
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--limit N` | Evaluate only the first `N` videos. Useful for smoke tests. |
| `--max-frames N` | Evaluate only the first `N` frames per video. |
| `--batch-size N` | Metric inference batch size. Lower this if GPU memory is tight. |
| `--overwrite-metrics` | Recompute outputs that already have `metrics.json`. |
| `--overwrite-crops` | Recreate existing `video_generated_bottom.mp4` files. |
| `--keep-going` | Continue after a failed output and write a failed `metrics.json`. |
| `--device cuda` | Force CUDA. Use `--device cpu` if needed, but it is much slower. |

## Output Files

For each rollout directory, the evaluator writes:

```text
video.mp4
video_generated_bottom.mp4
metrics.json
```

It also writes a sweep summary, usually:

```text
/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep/metrics_summary.json
```

Each per-output `metrics.json` has this shape:

```json
{
  "status": "ok",
  "source_video": ".../video.mp4",
  "cropped_video": ".../video_generated_bottom.mp4",
  "metrics": {
    "niqe": 5.33,
    "musiq": 47.12,
    "clipiqa": 0.22
  },
  "result": {
    "num_frames": 477,
    "pred_final_resolution": [704, 1280],
    "niqe_values": [5.1, 5.4],
    "musiq_values": [48.0, 47.8],
    "clipiqa_values": [0.21, 0.22]
  },
  "config": {
    "metrics": ["niqe", "musiq", "clipiqa"],
    "batch_size": 16
  }
}
```

The top-level `metrics` values are means across evaluated frames. The
`*_values` arrays in `result` contain per-frame scores.

If an output fails and `--keep-going` is set, its `metrics.json` contains:

```json
{
  "status": "failed",
  "error": {
    "type": "RuntimeError",
    "message": "..."
  }
}
```

## Interpreting Results

Use the metrics for relative comparisons inside the same sweep:

- Prefer lower `niqe`.
- Prefer higher `musiq`.
- Prefer higher `clipiqa`.
- Compare outputs only when they use the same crop, resolution, frame count, and metric set.
- Look for agreement across metrics. A sample with better `musiq` but worse `niqe` should be inspected visually.
- Compare seed triplets for stability. Large score swings across seeds may indicate unstable generation even if one seed scores well.
- Treat no-reference metrics as quality proxies. They can penalize unusual but valid scenes, weather, lighting, or motion patterns.

For the completed quality sweep, the aggregate over 96 outputs was:

| Metric | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `niqe` | 5.3353 | 1.1493 | 2.5779 | 7.6086 |
| `musiq` | 47.1285 | 8.7345 | 29.5497 | 59.6210 |
| `clipiqa` | 0.2269 | 0.0560 | 0.1503 | 0.3525 |

Use these as a baseline for future sweeps with the same evaluation settings.

## Quick Checks

Count completed artifacts:

```bash
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name video.mp4 | wc -l
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name video_generated_bottom.mp4 | wc -l
find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep -name metrics.json | wc -l
```

Check crop dimensions on one output:

```bash
sample=$(find /home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep \
  -name video_generated_bottom.mp4 | head -n 1)
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration \
  -of default=nokey=1:noprint_wrappers=1 "$sample"
```

Summarize all per-output metrics:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep")
vals = {"niqe": [], "musiq": [], "clipiqa": []}

for path in root.rglob("metrics.json"):
    data = json.loads(path.read_text())
    if data.get("status") != "ok":
        continue
    for name in vals:
        vals[name].append(data["metrics"][name])

for name, xs in vals.items():
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    print(
        name,
        "count", len(xs),
        "mean", f"{mean:.4f}",
        "std", f"{var ** 0.5:.4f}",
        "min", f"{min(xs):.4f}",
        "max", f"{max(xs):.4f}",
    )
PY
```
