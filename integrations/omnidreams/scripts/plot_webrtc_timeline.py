#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualise WebRTC pipeline profiling events as an interactive HTML timeline.

No external dependencies — uses only Python stdlib.  Output is a self-contained
HTML file with inline CSS/JS that you can open in any browser.

Usage:
    # 1. Run the demo with profiling enabled:
    WEBRTC_PROFILE=1 uv run --package flashdreams-omnidreams torchrun ... -m omnidreams.webrtc.server ...

    # 2. After the session, generate the timeline:
    python3 scripts/plot_webrtc_timeline.py /tmp/webrtc_profile.jsonl

    # Optionally limit the time window (seconds from epoch):
    python3 scripts/plot_webrtc_timeline.py /tmp/webrtc_profile.jsonl --start 5 --end 15

    # Custom output path:
    python3 scripts/plot_webrtc_timeline.py /tmp/webrtc_profile.jsonl -o my_timeline.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

STAGE_COLORS = {
    "trigger_sleep":               "#bdbdbd",
    "chunk_inference":             "#1f77b4",
    "prebuffer.inference":         "#1f77b4",
    "pose_integration":            "#aec7e8",
    "start_generation":            "#2ca02c",
    "continue_generation":         "#2ca02c",
    "finalize_kv_cache":           "#ff7f0e",
    "nvenc_encode":                "#d62728",
    "abgr_conversion":             "#e377c2",
    "encode_and_deliver.encode":   "#d62728",
    "encode_and_deliver.enqueue":  "#9467bd",
    "prebuffer.encode":            "#d62728",
    "recv.queue_wait":             "#8c564b",
    "recv.pacing_sleep":           "#7f7f7f",
}

STAGE_ROWS = {
    "trigger_sleep":               0,
    "chunk_inference":             1,
    "prebuffer.inference":         1,
    "pose_integration":            2,
    "start_generation":            3,
    "continue_generation":         3,
    "finalize_kv_cache":           4,
    "encode_and_deliver.encode":   5,
    "nvenc_encode":                6,
    "abgr_conversion":             6,
    "prebuffer.encode":            5,
    "encode_and_deliver.enqueue":  7,
    "recv.queue_wait":             8,
    "recv.pacing_sleep":           9,
}

ROW_LABELS = {
    0: "trigger_sleep",
    1: "chunk_inference",
    2: "pose_integration",
    3: "model_generation",
    4: "finalize_kv",
    5: "encode_task",
    6: "nvenc / abgr",
    7: "enqueue",
    8: "recv.queue_wait",
    9: "recv.pacing",
}


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def build_summary(events: list[dict]) -> str:
    by_stage: dict[str, list[float]] = defaultdict(list)
    for e in events:
        by_stage[e["stage"]].append(e["dur_ms"])
    lines = []
    lines.append(f"{'Stage':<35s} {'Count':>6s} {'Mean ms':>9s} {'Min ms':>9s} {'Max ms':>9s} {'Std ms':>9s}")
    lines.append("-" * 80)
    for stage in sorted(by_stage.keys()):
        vals = by_stage[stage]
        n = len(vals)
        mean = sum(vals) / n
        mn = min(vals)
        mx = max(vals)
        variance = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        std = variance ** 0.5
        lines.append(f"{stage:<35s} {n:>6d} {mean:>9.2f} {mn:>9.2f} {mx:>9.2f} {std:>9.2f}")
    return "\n".join(lines)


def generate_html(events: list[dict], t_min: float, t_max: float) -> str:
    filtered = [e for e in events if e["end"] >= t_min and e["start"] <= t_max]

    all_rows = sorted(ROW_LABELS.keys())
    n_rows = len(all_rows)

    events_json = json.dumps(filtered)
    stage_colors_json = json.dumps(STAGE_COLORS)
    stage_rows_json = json.dumps(STAGE_ROWS)
    row_labels_json = json.dumps(ROW_LABELS)

    summary_text = html.escape(build_summary(events))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WebRTC Pipeline Timeline</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
       background: #1a1a2e; color: #e0e0e0; }}
#header {{ padding: 12px 20px; background: #16213e; border-bottom: 1px solid #334; }}
#header h1 {{ font-size: 18px; font-weight: 600; }}
#header .info {{ font-size: 12px; color: #888; margin-top: 4px; }}
#controls {{ padding: 8px 20px; background: #1a1a2e; border-bottom: 1px solid #222;
             display: flex; gap: 16px; align-items: center; font-size: 13px; }}
#controls label {{ color: #aaa; }}
#controls input {{ width: 80px; background: #222; color: #eee; border: 1px solid #444;
                   border-radius: 3px; padding: 2px 6px; font-size: 13px; font-family: monospace; }}
#controls button {{ background: #2a4a7f; color: #ddd; border: 1px solid #446;
                    border-radius: 3px; padding: 3px 12px; cursor: pointer; font-size: 13px; }}
#controls button:hover {{ background: #3a5a9f; }}
#canvas-container {{ position: relative; overflow: hidden; }}
canvas {{ display: block; }}
#tooltip {{ position: absolute; display: none; background: #222; color: #eee;
            padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none;
            white-space: pre; border: 1px solid #555; z-index: 10; max-width: 400px; }}
#summary {{ padding: 16px 20px; background: #16213e; border-top: 1px solid #334; }}
#summary h2 {{ font-size: 14px; margin-bottom: 8px; }}
#summary pre {{ font-size: 12px; color: #bbb; overflow-x: auto; }}
#legend {{ padding: 8px 20px; background: #1a1a2e; display: flex; flex-wrap: wrap; gap: 8px 16px;
           font-size: 11px; border-bottom: 1px solid #222; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 14px; height: 10px; border-radius: 2px; }}
</style>
</head>
<body>
<div id="header">
  <h1>WebRTC Pipeline Timeline</h1>
  <div class="info" id="info-line"></div>
</div>
<div id="legend"></div>
<div id="controls">
  <label>Start(s): <input id="inp-start" type="number" step="0.1"></label>
  <label>End(s): <input id="inp-end" type="number" step="0.1"></label>
  <button id="btn-apply">Apply</button>
  <button id="btn-reset">Reset</button>
  <span style="color:#666; font-size:11px">Scroll to zoom · Drag to pan</span>
</div>
<div id="canvas-container">
  <canvas id="timeline"></canvas>
  <div id="tooltip"></div>
</div>
<div id="summary">
  <h2>Per-Stage Summary</h2>
  <pre>{summary_text}</pre>
</div>

<script>
const ALL_EVENTS = {events_json};
const STAGE_COLORS = {stage_colors_json};
const STAGE_ROWS = {stage_rows_json};
const ROW_LABELS = {row_labels_json};
const N_ROWS = {n_rows};
const ROW_HEIGHT = 32;
const LABEL_WIDTH = 140;
const TOP_PAD = 10;

const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const container = document.getElementById("canvas-container");

let viewStart = {t_min};
let viewEnd = {t_max};
let dataStart = {t_min};
let dataEnd = {t_max};

function resize() {{
  canvas.width = container.clientWidth;
  canvas.height = N_ROWS * ROW_HEIGHT + TOP_PAD + 30;
  draw();
}}

function xForTime(t) {{
  return LABEL_WIDTH + (t - viewStart) / (viewEnd - viewStart) * (canvas.width - LABEL_WIDTH - 10);
}}

function timeForX(x) {{
  return viewStart + (x - LABEL_WIDTH) / (canvas.width - LABEL_WIDTH - 10) * (viewEnd - viewStart);
}}

function draw() {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // row backgrounds
  const rowKeys = Object.keys(ROW_LABELS).map(Number).sort((a,b) => a - b);
  for (const row of rowKeys) {{
    const y = TOP_PAD + row * ROW_HEIGHT;
    ctx.fillStyle = row % 2 === 0 ? "#1e1e36" : "#1a1a2e";
    ctx.fillRect(0, y, w, ROW_HEIGHT);
    // label
    ctx.fillStyle = "#999";
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(ROW_LABELS[row], LABEL_WIDTH - 8, y + ROW_HEIGHT / 2);
  }}

  // grid lines
  const span = viewEnd - viewStart;
  let gridStep = 0.001;
  const targets = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50];
  const pixPerSec = (w - LABEL_WIDTH - 10) / span;
  for (const t of targets) {{ if (t * pixPerSec >= 60) {{ gridStep = t; break; }} }}
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 0.5;
  ctx.font = "10px monospace";
  ctx.fillStyle = "#666";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  let gt = Math.ceil(viewStart / gridStep) * gridStep;
  while (gt <= viewEnd) {{
    const gx = xForTime(gt);
    ctx.beginPath(); ctx.moveTo(gx, TOP_PAD); ctx.lineTo(gx, h - 20); ctx.stroke();
    ctx.fillText(gt.toFixed(3) + "s", gx, h - 18);
    gt += gridStep;
  }}

  // events
  for (const ev of ALL_EVENTS) {{
    if (ev.end < viewStart || ev.start > viewEnd) continue;
    const row = STAGE_ROWS[ev.stage];
    if (row === undefined) continue;
    const x1 = Math.max(xForTime(ev.start), LABEL_WIDTH);
    const x2 = Math.min(xForTime(ev.end), w - 10);
    const y = TOP_PAD + row * ROW_HEIGHT + 4;
    const bh = ROW_HEIGHT - 8;
    const bw = Math.max(x2 - x1, 1);
    ctx.fillStyle = STAGE_COLORS[ev.stage] || "#17becf";
    ctx.globalAlpha = 0.88;
    ctx.fillRect(x1, y, bw, bh);
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 0.4;
    ctx.strokeRect(x1, y, bw, bh);
    // label if wide enough
    if (bw > 40) {{
      ctx.fillStyle = "#fff";
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const label = ev.dur_ms.toFixed(1) + "ms";
      ctx.fillText(label, x1 + 3, y + bh / 2);
    }}
  }}

  // separator
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(LABEL_WIDTH, 0); ctx.lineTo(LABEL_WIDTH, h); ctx.stroke();
}}

// Tooltip
canvas.addEventListener("mousemove", (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const t = timeForX(mx);
  const rowIdx = Math.floor((my - TOP_PAD) / ROW_HEIGHT);

  let hit = null;
  for (const ev of ALL_EVENTS) {{
    const r = STAGE_ROWS[ev.stage];
    if (r !== rowIdx) continue;
    if (t >= ev.start && t <= ev.end) {{ hit = ev; break; }}
  }}
  if (hit) {{
    let lines = [`stage: ${{hit.stage}}`, `dur: ${{hit.dur_ms.toFixed(3)}} ms`,
                 `start: ${{hit.start.toFixed(6)}}s`, `end: ${{hit.end.toFixed(6)}}s`,
                 `chunk: ${{hit.chunk}}`];
    if (hit.tid) lines.push(`thread: ${{hit.tid}}`);
    for (const [k, v] of Object.entries(hit)) {{
      if (!["stage","start","end","dur_ms","chunk","tid"].includes(k))
        lines.push(`${{k}}: ${{JSON.stringify(v)}}`);
    }}
    tooltip.textContent = lines.join("\\n");
    tooltip.style.display = "block";
    let tx = e.clientX - rect.left + 14;
    let ty = e.clientY - rect.top - 10;
    if (tx + 300 > canvas.width) tx = mx - 310;
    tooltip.style.left = tx + "px";
    tooltip.style.top = ty + "px";
  }} else {{
    tooltip.style.display = "none";
  }}
}});
canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

// Zoom (scroll)
canvas.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const pivot = timeForX(mx);
  const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
  let newStart = pivot - (pivot - viewStart) * factor;
  let newEnd = pivot + (viewEnd - pivot) * factor;
  if (newEnd - newStart < 0.001) return;
  viewStart = newStart;
  viewEnd = newEnd;
  draw();
}});

// Pan (drag)
let dragging = false, dragStartX = 0, dragViewStart = 0, dragViewEnd = 0;
canvas.addEventListener("mousedown", (e) => {{
  dragging = true; dragStartX = e.clientX;
  dragViewStart = viewStart; dragViewEnd = viewEnd;
  canvas.style.cursor = "grabbing";
}});
window.addEventListener("mousemove", (e) => {{
  if (!dragging) return;
  const dx = e.clientX - dragStartX;
  const dt = -dx / ((canvas.width - LABEL_WIDTH - 10) / (dragViewEnd - dragViewStart));
  viewStart = dragViewStart + dt;
  viewEnd = dragViewEnd + dt;
  draw();
}});
window.addEventListener("mouseup", () => {{
  dragging = false; canvas.style.cursor = "default";
}});

// Controls
document.getElementById("inp-start").value = dataStart.toFixed(1);
document.getElementById("inp-end").value = dataEnd.toFixed(1);
document.getElementById("btn-apply").addEventListener("click", () => {{
  const s = parseFloat(document.getElementById("inp-start").value);
  const e = parseFloat(document.getElementById("inp-end").value);
  if (!isNaN(s) && !isNaN(e) && e > s) {{ viewStart = s; viewEnd = e; draw(); }}
}});
document.getElementById("btn-reset").addEventListener("click", () => {{
  viewStart = dataStart; viewEnd = dataEnd; draw();
}});

// Legend
const legendEl = document.getElementById("legend");
const seen = new Set();
for (const [stage, color] of Object.entries(STAGE_COLORS)) {{
  if (seen.has(stage)) continue; seen.add(stage);
  const present = ALL_EVENTS.some(e => e.stage === stage);
  if (!present) continue;
  const item = document.createElement("div"); item.className = "legend-item";
  item.innerHTML = `<span class="legend-swatch" style="background:${{color}}"></span>${{stage}}`;
  legendEl.appendChild(item);
}}

// Info line
document.getElementById("info-line").textContent =
  `${{ALL_EVENTS.length}} events · ${{dataStart.toFixed(3)}}s – ${{dataEnd.toFixed(3)}}s`;

window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WebRTC pipeline timeline (HTML)")
    parser.add_argument("profile", type=Path, help="Path to webrtc_profile.jsonl")
    parser.add_argument("--start", type=float, default=None, help="Start time (s)")
    parser.add_argument("--end", type=float, default=None, help="End time (s)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output HTML path (default: <profile>.html)")
    args = parser.parse_args()

    events = load_events(args.profile)
    if not events:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(events)} events from {args.profile}")
    print()
    print(build_summary(events))
    print()

    t_min = args.start if args.start is not None else 0.0
    t_max = args.end if args.end is not None else max(e["end"] for e in events) + 0.1

    out_path = args.output or args.profile.with_suffix(".html")
    html_content = generate_html(events, t_min, t_max)
    out_path.write_text(html_content)
    print(f"Timeline written to {out_path}")
    print(f"Open in browser:  file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
