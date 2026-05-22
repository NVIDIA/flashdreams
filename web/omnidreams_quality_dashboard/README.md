# OmniDreams Quality Dashboard

Static dashboard for browsing OmniDreams quality-sweep outputs.

Serve it from the repository root so the app can fetch both the dashboard files
and the sweep outputs:

```bash
cd /home/gtong/github/flashdreams
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/
```

By default the app reads:

```text
/outputs/omnidreams-quality-sweep/metrics_summary.json
```

To point at another summary file served by the same local server:

```text
http://127.0.0.1:8765/web/omnidreams_quality_dashboard/?summary=/outputs/another-sweep/metrics_summary.json
```
