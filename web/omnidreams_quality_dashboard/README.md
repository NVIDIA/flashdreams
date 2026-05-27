# OmniDreams Quality Dashboard

Static dashboard for browsing OmniDreams quality-sweep outputs.

Use the consolidated runbook for data prep, batch generation, metric evaluation,
VLM artifact evaluation, and dashboard inspection:

```text
docs/omnidreams_quality_sweep_runbook.md
```

Quick local launch from the repository root:

```bash
export FLASHDREAMS_REPO=/path/to/flashdreams
cd "$FLASHDREAMS_REPO"
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/
```

The dashboard can load explicit summary files:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?summary=/outputs/another-sweep/metrics_summary.json&vlm=/outputs/another-sweep/vlm_artifacts_summary.json
```

VLM rows with `response_valid: false` are shown as schema warnings and count as
needing review, even if numeric artifact severities are zero.
