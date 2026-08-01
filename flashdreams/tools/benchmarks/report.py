# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTML report generation for local benchmark runs."""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricDisplay:
    """Presentation metadata for a normalized metric key."""

    label: str
    unit: str = ""
    scale: float = 1.0
    chart_kind: str = "other"


@dataclass(frozen=True)
class ModelReportGroup:
    """Scenarios grouped into one per-model detail report."""

    slug: str
    label: str
    scenarios: tuple[dict[str, Any], ...]


_LABEL_OVERRIDES = {
    "command_wall_s": "Command wall time",
    "total_s": "Total",
    "total_wo_finalize_s": "Total without finalize",
    "diffuse_s": "Diffuse",
    "encode_s": "Encode",
    "decode_s": "Decode",
    "finalize_s": "Finalize",
    "generate_s": "Generate",
    "model_step_s": "Model step",
    "cache_seed_prune_s": "Cache seed/prune",
    "gpu_to_cpu_copy_s": "GPU to CPU copy",
    "mem_alloc_gib": "Allocated memory",
    "mem_peak_gib": "Peak memory",
    "mem_reserved_gib": "Reserved memory",
    "pixel_fps": "Pixel throughput",
    "wall_present_fps": "Presented frame rate",
    "quality_score": "Quality score",
    "quality_similarity_score": "Clip similarity score",
    "quality_visual_sanity_score": "Visual sanity score",
    "quality_temporal_score": "Temporal stability score",
    "quality_ssim_score": "SSIM similarity score",
    "quality_flip_score": "FLIP similarity score",
    "quality_mean_abs": "Mean absolute difference",
    "quality_rmse": "RMSE",
    "quality_psnr_db": "PSNR",
    "quality_max_frame_mean_abs": "Worst-frame mean abs",
    "quality_max_frame_rmse": "Worst-frame RMSE",
    "quality_mean_flip": "Mean FLIP",
    "quality_max_frame_flip": "Worst-frame FLIP",
    "quality_reference_frame_count": "Reference frames",
    "quality_candidate_frame_count": "Candidate frames",
    "quality_sampled_frame_count": "Sampled frames",
    "pai_bench_g_score": "PAI-Bench-G score",
    "pai_bench_long_score": "PAI-Bench-Long score",
    "pai_bench_g_dimensions_evaluated": "PAI-Bench-G dimensions",
    "pai_bench_long_dimensions_evaluated": "PAI-Bench-Long dimensions",
    "pai_bench_g_videos_evaluated": "PAI-Bench-G videos",
    "pai_bench_long_videos_evaluated": "PAI-Bench-Long videos",
    "returncode": "Return code",
}
_CHART_ORDER = {
    "total_s": 0,
    "total_wo_finalize_s": 1,
    "diffuse_s": 2,
    "encode_s": 3,
    "decode_s": 4,
    "finalize_s": 5,
    "generate_s": 6,
    "model_step_s": 7,
    "cache_seed_prune_s": 8,
    "gpu_to_cpu_copy_s": 9,
    "mem_peak_gib": 0,
    "mem_alloc_gib": 1,
    "mem_reserved_gib": 2,
    "pixel_fps": 0,
    "wall_present_fps": 1,
    "quality_score": 0,
    "quality_similarity_score": 1,
    "quality_visual_sanity_score": 2,
    "quality_temporal_score": 3,
    "quality_ssim_score": 4,
    "quality_flip_score": 5,
    "quality_rmse": 0,
    "quality_mean_abs": 1,
    "quality_psnr_db": 2,
    "pai_bench_long_score": 0,
    "pai_bench_g_score": 1,
}
_DERIVED_SUMMARY_SUFFIXES = (
    ("_median_s", "median"),
    ("_p90_s", "p90"),
    ("_mean_s", "mean"),
    ("_min_s", "min"),
    ("_max_s", "max"),
    ("_median_fps", "median"),
    ("_p90_fps", "p90"),
    ("_mean_fps", "mean"),
    ("_min_fps", "min"),
    ("_max_fps", "max"),
)


def write_html_report(manifest: dict[str, Any], path: Path) -> Path:
    """Write an index report plus per-model detail pages."""
    manifest_output_root = Path(str(manifest.get("output_root", path.parent)))
    asset_root = manifest_output_root if manifest_output_root.exists() else path.parent
    scenarios = [
        scenario
        for scenario in manifest.get("scenarios", [])
        if isinstance(scenario, dict)
    ]
    groups = _model_report_groups(scenarios)
    detail_dir = path.parent / "reports"
    detail_hrefs_by_slug: dict[str, str] = {}
    scenario_detail_hrefs: dict[str, str] = {}
    for group in groups:
        detail_path = detail_dir / f"{group.slug}.html"
        detail_href = _file_href(detail_path, page_dir=path.parent)
        detail_hrefs_by_slug[group.slug] = detail_href
        for scenario in group.scenarios:
            scenario_id = str(scenario.get("id", ""))
            scenario_detail_hrefs[scenario_id] = (
                f"{detail_href}#{_scenario_anchor(scenario_id)}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        detail_path = detail_dir / f"{group.slug}.html"
        detail_path.write_text(
            _html_document(
                f"{group.label} Benchmark Report",
                _model_detail_body(
                    manifest,
                    group=group,
                    manifest_output_root=manifest_output_root,
                    asset_root=asset_root,
                    page_dir=detail_path.parent,
                    summary_href=_file_href(path, page_dir=detail_path.parent),
                ),
            ),
            encoding="utf-8",
        )

    path.write_text(
        _html_document(
            "FlashDreams Benchmark Report",
            _summary_body(
                manifest,
                scenarios=scenarios,
                groups=groups,
                detail_hrefs_by_slug=detail_hrefs_by_slug,
                scenario_detail_hrefs=scenario_detail_hrefs,
            ),
        ),
        encoding="utf-8",
    )
    return path


def _html_document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
{_report_css()}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _summary_body(
    manifest: dict[str, Any],
    *,
    scenarios: list[dict[str, Any]],
    groups: list[ModelReportGroup],
    detail_hrefs_by_slug: Mapping[str, str],
    scenario_detail_hrefs: Mapping[str, str],
) -> str:
    environment = manifest.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    quality_baseline = manifest.get("quality_baseline")
    return f"""
  <h1>FlashDreams Benchmark Report</h1>
  <p class="muted">Created {html.escape(str(manifest.get("created_at", "")))}.</p>

  <h2>Run</h2>
  {_run_table(manifest, environment=environment, quality_baseline=quality_baseline)}

  <h2>Model Reports</h2>
  {_model_report_cards(groups, detail_hrefs_by_slug) or _empty_model_reports()}

  <h2>Scenario Highlights</h2>
  {_scenario_highlights(scenarios, detail_hrefs=scenario_detail_hrefs) or _empty_highlights()}

  <h2>Scenarios</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Scenario</th>
        <th>Status</th>
        <th>Wall Time</th>
        <th>Quality</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {_scenario_index_rows(groups, detail_hrefs_by_slug)}
    </tbody>
  </table>

  <h2>Quality Guide</h2>
  {_quality_guide()}
"""


def _model_detail_body(
    manifest: dict[str, Any],
    *,
    group: ModelReportGroup,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
    summary_href: str,
) -> str:
    scenarios = list(group.scenarios)
    scenario_rows = "\n".join(
        _scenario_row(
            scenario,
            manifest_output_root=manifest_output_root,
            asset_root=asset_root,
            page_dir=page_dir,
        )
        for scenario in scenarios
    )
    chart_sections = _metric_charts(scenarios)
    metric_rows = "\n".join(_metric_summary_rows(scenario) for scenario in scenarios)
    highlight_sections = _scenario_highlights(scenarios)
    quality_comparisons = _quality_comparison_sections(
        scenarios,
        manifest_output_root=manifest_output_root,
        asset_root=asset_root,
        page_dir=page_dir,
    )
    created_at = html.escape(str(manifest.get("created_at", "")))
    return f"""
  <nav class="top-nav"><a href="{html.escape(summary_href)}">Back to summary</a></nav>
  <h1>{html.escape(group.label)} Benchmark Report</h1>
  <p class="muted">Created {created_at}. {len(scenarios)} scenario(s).</p>

  <h2>Scenario Highlights</h2>
  {highlight_sections or _empty_highlights()}

  <h2>Scenarios</h2>
  <div class="table-scroll">
  <table class="scenario-table">
    <colgroup>
      <col class="scenario-col">
      <col class="status-col">
      <col class="wall-col">
      <col class="command-col">
      <col class="artifacts-col">
    </colgroup>
    <thead>
      <tr>
        <th>Scenario</th>
        <th>Status</th>
        <th>Wall Time</th>
        <th>Command</th>
        <th>Artifacts</th>
      </tr>
    </thead>
    <tbody>
      {scenario_rows}
    </tbody>
  </table>
  </div>

  <h2>Quality Guide</h2>
  {_quality_guide()}

  <h2>Quality Comparisons</h2>
  {quality_comparisons or _empty_quality_comparisons()}

  <h2>Metric Charts</h2>
  {chart_sections or _empty_charts()}

  <h2>Metric Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Scenario</th><th>Metric</th><th>Unit</th><th>Count</th>
        <th>Median</th>
      </tr>
    </thead>
    <tbody>
      {metric_rows or _empty_metrics_row()}
    </tbody>
  </table>
"""


def _report_css() -> str:
    return """
    body {
      margin: 32px;
      color: #161616;
      background: #fafafa;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3 { margin: 0 0 12px; }
    h2 { margin-top: 28px; }
    h3 { margin-top: 18px; }
    a { color: #174ea6; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #f1f3f4; }
    td, th { overflow-wrap: anywhere; }
    code {
      background: #f0f0f0;
      padding: 1px 4px;
      border-radius: 3px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .numeric { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .top-nav { margin-bottom: 18px; }
    .top-nav a { font-weight: 600; }
    .detail-link { font-weight: 600; white-space: nowrap; }
    .status-pass { color: #0b6e2b; font-weight: 600; }
    .status-fail, .status-timeout { color: #a32020; font-weight: 600; }
    .status-dry_run { color: #555; font-weight: 600; }
    .muted { color: #5f6368; }
    .highlight-grid, .comparison-grid, .model-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
      gap: 14px;
      margin-bottom: 12px;
    }
    .highlight-card, .guide-card, .comparison-card, .model-card {
      min-width: 0;
      background: white;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 12px;
    }
    .highlight-card h3, .guide-card h3, .comparison-card h4, .model-card h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .model-card p { margin: 6px 0; }
    .highlight-list {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 6px 12px;
      margin: 0;
    }
    .highlight-list dt { color: #5f6368; }
    .highlight-list dd { margin: 0; font-variant-numeric: tabular-nums; }
    .highlight-list strong { font-size: 16px; }
    .metric-help { display: block; margin-top: 3px; color: #5f6368; }
    .metric-pill {
      display: inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .metric-pill.good { color: #0b6e2b; background: #e7f4ea; }
    .metric-pill.warn { color: #8a5a00; background: #fff4d8; }
    .metric-pill.bad { color: #a32020; background: #fce8e6; }
    .metric-pill.neutral { color: #3c4043; background: #f1f3f4; }
    .quality-score-list {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 6px 12px;
      margin: 10px 0 0;
    }
    .quality-score-list dt { color: #5f6368; }
    .quality-score-list dd { margin: 0; font-variant-numeric: tabular-nums; }
    .quality-guide-table td:first-child { white-space: nowrap; }
    .table-scroll {
      overflow-x: auto;
      margin-bottom: 12px;
    }
    .scenario-table {
      table-layout: fixed;
      min-width: 900px;
    }
    .scenario-table .scenario-col { width: 26%; }
    .scenario-table .status-col { width: 8%; }
    .scenario-table .wall-col { width: 10%; }
    .scenario-table .command-col { width: 16%; }
    .scenario-table .artifacts-col { width: 40%; }
    .scenario-table td { min-width: 0; }
    .scenario-cell strong { overflow-wrap: anywhere; }
    .wall-time-cell { white-space: nowrap; }
    .command-details summary {
      cursor: pointer;
      color: #174ea6;
      font-weight: 600;
      white-space: nowrap;
    }
    .command-text {
      display: block;
      max-width: 100%;
      max-height: 10rem;
      margin-top: 6px;
      overflow: auto;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
      gap: 14px;
      margin-bottom: 10px;
    }
    .chart-card {
      min-width: 0;
      background: white;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 12px;
    }
    .chart-card h4 { margin: 0 0 8px; font-size: 14px; }
    .chart-card table { background: transparent; }
    .chart-card th, .chart-card td { border: 0; border-bottom: 1px solid #eee; padding: 6px 4px; }
    .chart-card tr:last-child td { border-bottom: 0; }
    .value-bar {
      display: grid;
      grid-template-columns: minmax(70px, 1fr) auto;
      gap: 8px;
      align-items: center;
      min-width: 150px;
    }
    .bar-track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e8edf3;
    }
    .bar-fill { display: block; height: 100%; background: #2b6cb0; }
    .bar-fill.secondary { background: #b7791f; }
    .chart-value { font-variant-numeric: tabular-nums; white-space: nowrap; }
    video {
      width: min(100%, 320px);
      height: auto;
      max-height: 180px;
      display: block;
      margin-top: 6px;
    }
    .comparison-card a { overflow-wrap: anywhere; }
    .comparison-card video { width: 100%; max-width: 520px; max-height: 300px; }
    @media (max-width: 700px) {
      body { margin: 16px; }
      th, td { padding: 6px; }
      .highlight-list, .quality-score-list { grid-template-columns: 1fr; }
      .metric-pill { margin-left: 0; margin-top: 3px; }
    }
    """


def _run_table(
    manifest: dict[str, Any],
    *,
    environment: dict[str, Any],
    quality_baseline: object,
) -> str:
    return f"""
  <table>
    <tbody>
      <tr><th>Mode</th><td>{html.escape(str(manifest.get("mode", "")))}</td></tr>
      <tr><th>Output Root</th><td><code>{_escape(manifest.get("output_root", ""))}</code></td></tr>
      <tr><th>Quality Baseline</th><td>{_quality_baseline_summary(quality_baseline)}</td></tr>
      <tr>
        <th>Git Commit</th>
        <td><code>{_escape(_dig(environment, "git", "commit") or "")}</code></td>
      </tr>
      <tr><th>Git Branch</th><td>{_escape(_dig(environment, "git", "branch") or "")}</td></tr>
      <tr><th>Dirty</th><td>{html.escape(str(_dig(environment, "git", "dirty") or ""))}</td></tr>
      <tr><th>GPU</th><td>{_gpu_summary(environment)}</td></tr>
      <tr>
        <th>Python</th>
        <td>{_escape(_first_line(_dig(environment, "python", "version")))}</td>
      </tr>
      <tr><th>Torch</th><td>{_escape(_dig(environment, "torch", "version") or "")}</td></tr>
    </tbody>
  </table>
"""


def _model_report_groups(scenarios: list[dict[str, Any]]) -> list[ModelReportGroup]:
    by_slug: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    order: list[str] = []
    for scenario in scenarios:
        slug, label = _scenario_group(scenario)
        if slug not in by_slug:
            by_slug[slug] = (label, [])
            order.append(slug)
        by_slug[slug][1].append(scenario)
    return [
        ModelReportGroup(
            slug=slug,
            label=by_slug[slug][0],
            scenarios=tuple(by_slug[slug][1]),
        )
        for slug in order
    ]


def _scenario_group(scenario: dict[str, Any]) -> tuple[str, str]:
    configured = _configured_report_group(scenario)
    if configured is not None:
        return configured
    scenario_id = str(scenario.get("id", ""))
    fallback = scenario_id.split("-", 1)[0] if scenario_id else "other"
    slug = _slugify(fallback)
    return slug, _title_metric(slug.replace("-", "_"))


def _configured_report_group(scenario: dict[str, Any]) -> tuple[str, str] | None:
    for value in (
        scenario.get("report_group"),
        _nested_scenario_value(scenario, "report_group"),
    ):
        parsed = _report_group_value(value)
        if parsed is not None:
            return parsed
    return None


def _nested_scenario_value(scenario: dict[str, Any], key: str) -> object:
    nested = scenario.get("scenario")
    if not isinstance(nested, dict):
        return None
    return nested.get(key)


def _report_group_value(value: object) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        slug = _slugify(value)
        return slug, _title_metric(slug.replace("-", "_"))
    if not isinstance(value, dict):
        return None
    group_id = value.get("id")
    if not isinstance(group_id, str) or not group_id:
        return None
    slug = _slugify(group_id)
    label = value.get("name")
    if isinstance(label, str) and label:
        return slug, label
    return slug, _title_metric(slug.replace("-", "_"))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "other"


def _scenario_anchor(scenario_id: str) -> str:
    return f"scenario-{_slugify(scenario_id)}"


def _status_class(status: str) -> str:
    css_class = re.sub(r"[^a-z0-9_-]+", "-", status.lower()).strip("-")
    return css_class or "unknown"


def _model_report_cards(
    groups: list[ModelReportGroup],
    detail_hrefs_by_slug: Mapping[str, str],
) -> str:
    cards: list[str] = []
    for group in groups:
        href = detail_hrefs_by_slug.get(group.slug, f"reports/{group.slug}.html")
        row_values = [
            ("Scenarios", str(len(group.scenarios))),
            ("Status", _status_summary(group.scenarios)),
        ]
        wall_time_s = _group_metric_value(group.scenarios, "command_wall_s")
        if wall_time_s is not None:
            row_values.append(("Median wall time", _format_wall_time(wall_time_s)))
        quality_score = _group_metric_value(group.scenarios, "quality_score")
        if quality_score is not None:
            display = _metric_display("quality_score")
            row_values.append(
                (
                    "Median quality",
                    f"{_format_metric_with_unit(quality_score, display)}"
                    f"{_metric_judgement_html('quality_score', quality_score)}",
                )
            )
        pai_score = _group_metric_value(group.scenarios, "pai_bench_long_score")
        if pai_score is not None:
            display = _metric_display("pai_bench_long_score")
            row_values.append(
                (
                    "Median PAI-Bench-Long",
                    f"{_format_metric_with_unit(pai_score, display)}"
                    f"{_metric_judgement_html('pai_bench_long_score', pai_score)}",
                )
            )
        rows = "".join(
            f"<dt>{html.escape(label)}</dt><dd>{value}</dd>"
            for label, value in row_values
        )
        scenario_list = ", ".join(
            html.escape(str(scenario.get("id", ""))) for scenario in group.scenarios
        )
        cards.append(
            '<div class="model-card">'
            f'<h3><a href="{html.escape(href)}">{html.escape(group.label)}</a></h3>'
            f'<dl class="highlight-list">{rows}</dl>'
            f'<p class="muted">{scenario_list}</p>'
            f'<p><a class="detail-link" href="{html.escape(href)}">View details</a></p>'
            "</div>"
        )
    return f'<div class="model-grid">{"".join(cards)}</div>' if cards else ""


def _scenario_index_rows(
    groups: list[ModelReportGroup],
    detail_hrefs_by_slug: Mapping[str, str],
) -> str:
    rows: list[str] = []
    for group in groups:
        detail_href = detail_hrefs_by_slug.get(group.slug, f"reports/{group.slug}.html")
        for scenario in group.scenarios:
            scenario_id = str(scenario.get("id", ""))
            scenario_href = f"{detail_href}#{_scenario_anchor(scenario_id)}"
            status = str(scenario.get("status", ""))
            rows.append(
                "<tr>"
                f'<td><a href="{html.escape(detail_href)}">{html.escape(group.label)}</a></td>'
                f'<td><a href="{html.escape(scenario_href)}">{html.escape(scenario_id)}</a><br>'
                f'<span class="muted">{html.escape(str(scenario.get("name", "")))}</span></td>'
                f'<td class="status-{_status_class(status)}">{html.escape(status)}</td>'
                f"<td>{_format_wall_time(scenario.get('wall_time_s'))}</td>"
                f"<td>{_scenario_quality_summary(scenario)}</td>"
                f'<td><a class="detail-link" href="{html.escape(scenario_href)}">Open</a></td>'
                "</tr>"
            )
    if not rows:
        return '<tr><td colspan="6" class="muted">No scenarios were run.</td></tr>'
    return "\n".join(rows)


def _scenario_quality_summary(scenario: dict[str, Any]) -> str:
    parts: list[str] = []
    for metric in (
        "quality_score",
        "quality_similarity_score",
        "pai_bench_long_score",
    ):
        value = _scenario_metric_value(scenario, metric)
        if value is None:
            continue
        display = _metric_display(metric)
        parts.append(
            f"{html.escape(display.label)}: "
            f"{_format_metric_with_unit(value, display)}"
            f"{_metric_judgement_html(metric, value)}"
        )
    return "<br>".join(parts) if parts else '<span class="muted">not available</span>'


def _status_summary(scenarios: tuple[dict[str, Any], ...]) -> str:
    counts: dict[str, int] = {}
    for scenario in scenarios:
        status = str(scenario.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(
        f'<span class="status-{_status_class(status)}">{html.escape(status)}</span> '
        f"{count}"
        for status, count in sorted(counts.items())
    )


def _group_metric_value(
    scenarios: tuple[dict[str, Any], ...],
    metric: str,
) -> float | None:
    values = [
        float(value)
        for scenario in scenarios
        if (value := _scenario_metric_value(scenario, metric)) is not None
    ]
    return _median(values)


def _scenario_metric_value(scenario: dict[str, Any], metric: str) -> float | int | None:
    highlights = scenario.get("metric_highlights", {})
    if not isinstance(highlights, dict):
        highlights = {}
    summary = scenario.get("metric_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    if metric == "command_wall_s":
        return _numeric_or_none(
            highlights.get("command_wall_s"),
            fallback=scenario.get("wall_time_s"),
        )
    return _numeric_or_none(
        highlights.get(f"{metric}_median"),
        fallback=_summary_stat(summary, metric, "median"),
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _scenario_row(
    scenario: dict[str, Any],
    *,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
) -> str:
    scenario_id = str(scenario.get("id", ""))
    status = str(scenario.get("status", ""))
    wall_time = _format_wall_time(scenario.get("wall_time_s"))
    command = _command_details(scenario.get("command", ""))
    artifact_links = _artifact_links(
        scenario,
        manifest_output_root=manifest_output_root,
        asset_root=asset_root,
        page_dir=page_dir,
    )
    return (
        "<tr>"
        f'<td class="scenario-cell" id="{html.escape(_scenario_anchor(scenario_id))}">'
        f"<strong>{html.escape(scenario_id)}</strong><br>"
        f'<span class="muted">{html.escape(str(scenario.get("name", "")))}</span></td>'
        f'<td class="status-{_status_class(status)}">{html.escape(status)}</td>'
        f'<td class="wall-time-cell">{wall_time}</td>'
        f'<td class="command-cell">{command}</td>'
        f'<td class="artifact-cell">{artifact_links}</td>'
        "</tr>"
    )


def _command_details(command: object) -> str:
    command_text = "" if command is None else str(command)
    if not command_text:
        return '<span class="muted">not recorded</span>'
    return (
        '<details class="command-details">'
        "<summary>Show command</summary>"
        f'<code class="command-text">{html.escape(command_text)}</code>'
        "</details>"
    )


def _artifact_links(
    scenario: dict[str, Any],
    *,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
) -> str:
    artifacts = scenario.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return '<span class="muted">none</span>'
    parts: list[str] = []
    videos = [str(path) for path in artifacts.get("videos", [])]
    for video in videos:
        href = html.escape(
            _artifact_href(
                video,
                manifest_output_root=manifest_output_root,
                asset_root=asset_root,
                page_dir=page_dir,
            )
        )
        parts.append(f'<a href="{href}">{html.escape(Path(video).name)}</a>')
        parts.append(f'<video controls preload="metadata" src="{href}"></video>')
    for kind in ("logs", "stats", "quality", "other"):
        for artifact in artifacts.get(kind, []):
            artifact_str = str(artifact)
            href = html.escape(
                _artifact_href(
                    artifact_str,
                    manifest_output_root=manifest_output_root,
                    asset_root=asset_root,
                    page_dir=page_dir,
                )
            )
            parts.append(f'<a href="{href}">{html.escape(Path(artifact_str).name)}</a>')
    return "<br>".join(parts) if parts else '<span class="muted">none</span>'


def _metric_summary_rows(scenario: dict[str, Any]) -> str:
    scenario_id = html.escape(str(scenario.get("id", "")))
    summary = scenario.get("metric_summary", {})
    if not isinstance(summary, dict):
        return ""
    summary = _display_metric_summary(
        summary,
        metadata=_metric_summary_metadata(scenario),
    )
    rows: list[str] = []
    for metric, stats in sorted(summary.items()):
        if not isinstance(stats, dict):
            continue
        display = _metric_display(str(metric))
        hint = _metric_hint(str(metric))
        judgement = _metric_judgement_html(str(metric), stats.get("median"))
        rows.append(
            "<tr>"
            f"<td>{scenario_id}</td>"
            f"<td>{html.escape(display.label)}<br>"
            f"<code>{html.escape(str(metric))}</code>{judgement}{hint}</td>"
            f"<td>{html.escape(display.unit or '')}</td>"
            f'<td class="numeric">{html.escape(str(stats.get("count", "")))}</td>'
            f'<td class="numeric">{_format_metric_value(stats.get("median"), display)}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _scenario_highlights(
    scenarios: list[dict[str, Any]],
    *,
    detail_hrefs: Mapping[str, str] | None = None,
) -> str:
    cards: list[str] = []
    for scenario in scenarios:
        scenario_id_raw = str(scenario.get("id", ""))
        scenario_id = html.escape(scenario_id_raw)
        detail_href = detail_hrefs.get(scenario_id_raw) if detail_hrefs else None
        scenario_heading = (
            f'<a href="{html.escape(detail_href)}">{scenario_id}</a>'
            if detail_href
            else scenario_id
        )
        status_raw = str(scenario.get("status", ""))
        status = html.escape(status_raw)
        status_class = _status_class(status_raw)
        highlights = scenario.get("metric_highlights", {})
        if not isinstance(highlights, dict):
            highlights = {}
        summary = scenario.get("metric_summary", {})
        if not isinstance(summary, dict):
            summary = {}

        command_wall_s = _numeric_or_none(
            highlights.get("command_wall_s"),
            fallback=scenario.get("wall_time_s"),
        )
        startup_step_total_s = _numeric_or_none(highlights.get("startup_step_total_s"))
        steady_total_s = _numeric_or_none(
            highlights.get("total_s_median"),
            fallback=_summary_stat(summary, "total_s", "median"),
        )
        quality_score = _numeric_or_none(
            highlights.get("quality_score_median"),
            fallback=_summary_stat(summary, "quality_score", "median"),
        )
        similarity_score = _numeric_or_none(
            highlights.get("quality_similarity_score_median"),
            fallback=_summary_stat(summary, "quality_similarity_score", "median"),
        )
        pai_bench_long_score = _numeric_or_none(
            highlights.get("pai_bench_long_score_median"),
            fallback=_summary_stat(summary, "pai_bench_long_score", "median"),
        )
        pai_bench_g_score = _numeric_or_none(
            highlights.get("pai_bench_g_score_median"),
            fallback=_summary_stat(summary, "pai_bench_g_score", "median"),
        )

        items = [
            (
                "Status",
                f'<span class="status-{status_class}">{status}</span>',
            )
        ]
        if command_wall_s is not None:
            items.append(
                (
                    "Command wall time",
                    f"<strong>{_format_wall_time(command_wall_s)}</strong><br>"
                    '<span class="muted">startup + generation + file writing</span>',
                )
            )
        if startup_step_total_s is not None:
            items.append(
                (
                    "Startup step",
                    f"<strong>{_format_wall_time(startup_step_total_s)}</strong><br>"
                    '<span class="muted">first recorded runner step/chunk</span>',
                )
            )
        if steady_total_s is not None:
            display = _metric_display("total_s")
            items.append(
                (
                    "Steady median step",
                    f"<strong>{_format_metric_with_unit(steady_total_s, display)}</strong>",
                )
            )
        if quality_score is not None:
            display = _metric_display("quality_score")
            items.append(
                (
                    "Quality score",
                    f"<strong>{_format_metric_with_unit(quality_score, display)}</strong>"
                    f"{_metric_judgement_html('quality_score', quality_score)}",
                )
            )
        if similarity_score is not None:
            display = _metric_display("quality_similarity_score")
            items.append(
                (
                    "Clip similarity",
                    f"<strong>{_format_metric_with_unit(similarity_score, display)}</strong>"
                    f"{_metric_judgement_html('quality_similarity_score', similarity_score)}",
                )
            )
        if pai_bench_long_score is not None:
            display = _metric_display("pai_bench_long_score")
            items.append(
                (
                    "PAI-Bench-Long",
                    f"<strong>{_format_metric_with_unit(pai_bench_long_score, display)}</strong>"
                    f"{_metric_judgement_html('pai_bench_long_score', pai_bench_long_score)}",
                )
            )
        if pai_bench_g_score is not None:
            display = _metric_display("pai_bench_g_score")
            items.append(
                (
                    "PAI-Bench-G",
                    f"<strong>{_format_metric_with_unit(pai_bench_g_score, display)}</strong>"
                    f"{_metric_judgement_html('pai_bench_g_score', pai_bench_g_score)}",
                )
            )

        rows = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in items)
        cards.append(
            '<div class="highlight-card">'
            f"<h3>{scenario_heading}</h3>"
            f'<dl class="highlight-list">{rows}</dl>'
            "</div>"
        )
    return f'<div class="highlight-grid">{"".join(cards)}</div>' if cards else ""


def _quality_guide() -> str:
    return """
      <div class="guide-card">
        <h3>How to read quality metrics</h3>
        <table class="quality-guide-table">
          <tbody>
            <tr>
              <td><code>quality_score</code></td>
              <td>Overall 0-1 summary. Higher is better. It blends baseline similarity with visual sanity checks.</td>
            </tr>
            <tr>
              <td><code>quality_similarity_score</code>, <code>quality_ssim_score</code></td>
              <td>0-1 closeness to the baseline MP4. Near 1.0 means visually close; below about 0.75 usually needs inspection.</td>
            </tr>
            <tr>
              <td><code>quality_rmse</code>, <code>quality_mean_abs</code></td>
              <td>8-bit pixel error. Lower is better; 0 is identical. As a rough guide, RMSE &lt; 15 or mean abs &lt; 8 is small, while RMSE &gt; 40 or mean abs &gt; 20 is large.</td>
            </tr>
            <tr>
              <td><code>quality_psnr_db</code></td>
              <td>Pixel similarity in dB. Higher is better; &gt;30 dB is usually close, 20-30 dB is visibly different, and &lt;20 dB is large drift.</td>
            </tr>
            <tr>
              <td><code>quality_visual_sanity_score</code>, <code>quality_temporal_score</code></td>
              <td>No-reference guardrails for blank, flat, striped, or unstable output. They catch obvious failures but do not prove semantic quality.</td>
            </tr>
            <tr>
              <td><code>pai_bench_long_score</code></td>
              <td>Optional 0-100 PAI-Bench-Long evaluator score for longer clips. Higher is better. It is an evaluator score, not a baseline similarity score.</td>
            </tr>
          </tbody>
        </table>
        <p class="muted">These bands are heuristics for local debugging, not CI gates. Compare runs with the same scenario, seed, inputs, and video region.</p>
      </div>
    """


def _quality_comparison_sections(
    scenarios: list[dict[str, Any]],
    *,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
) -> str:
    sections: list[str] = []
    for scenario in scenarios:
        scenario_id = html.escape(str(scenario.get("id", "")))
        quality_results = scenario.get("quality_results", [])
        if not isinstance(quality_results, list):
            continue
        for result in quality_results:
            if not isinstance(result, dict):
                continue
            result_id = str(result.get("id", ""))
            if result_id not in ("baseline-clip-compare", "baseline-video-review"):
                continue
            baseline_video = result.get("baseline_video")
            candidate_video = result.get("candidate_video")
            if not isinstance(baseline_video, str) or not isinstance(
                candidate_video,
                str,
            ):
                continue
            payload = _quality_payload(
                result,
                manifest_output_root=manifest_output_root,
                asset_root=asset_root,
            )
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            metrics = metrics if isinstance(metrics, dict) else {}
            metrics_card = (
                _manual_video_review_card(result)
                if result_id == "baseline-video-review"
                else _quality_metrics_card(metrics)
            )
            baseline_card = _comparison_video_card(
                "Baseline",
                baseline_video,
                manifest_output_root=manifest_output_root,
                asset_root=asset_root,
                page_dir=page_dir,
            )
            candidate_card = _comparison_video_card(
                "Candidate",
                candidate_video,
                manifest_output_root=manifest_output_root,
                asset_root=asset_root,
                page_dir=page_dir,
            )
            sections.append(
                "<section>"
                f"<h3>{scenario_id}</h3>"
                '<div class="comparison-grid">'
                f"{baseline_card}"
                f"{candidate_card}"
                f"{metrics_card}"
                "</div>"
                "</section>"
            )
    return "\n".join(sections)


def _comparison_video_card(
    label: str,
    path_value: str,
    *,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
) -> str:
    href = html.escape(
        _artifact_href(
            path_value,
            manifest_output_root=manifest_output_root,
            asset_root=asset_root,
            page_dir=page_dir,
        )
    )
    name = html.escape(Path(path_value).name)
    return (
        '<div class="comparison-card">'
        f"<h4>{html.escape(label)}</h4>"
        f'<a href="{href}">{name}</a>'
        f'<video controls preload="metadata" src="{href}"></video>'
        "</div>"
    )


def _quality_metrics_card(metrics: dict[str, Any]) -> str:
    preferred = (
        "quality_score",
        "quality_similarity_score",
        "quality_visual_sanity_score",
        "quality_temporal_score",
        "quality_ssim_score",
        "quality_rmse",
        "quality_mean_abs",
        "quality_psnr_db",
    )
    rows: list[str] = []
    for metric in preferred:
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        display = _metric_display(metric)
        rows.append(
            f"<dt>{html.escape(display.label)}</dt>"
            f"<dd>{_format_metric_with_unit(float(value), display)}"
            f"{_metric_judgement_html(metric, value)}</dd>"
        )
    if not rows:
        rows.append('<dt>Metrics</dt><dd><span class="muted">not available</span></dd>')
    return (
        '<div class="comparison-card">'
        "<h4>Quality metrics</h4>"
        f'<dl class="quality-score-list">{"".join(rows)}</dl>'
        "</div>"
    )


def _manual_video_review_card(result: dict[str, Any]) -> str:
    reason = html.escape(str(result.get("reason", "")))
    suffix = f"<br>{reason}" if reason else ""
    return (
        '<div class="comparison-card">'
        "<h4>Manual review</h4>"
        '<p class="muted">Baseline scoring was not run for this scenario.'
        f"{suffix}</p>"
        "</div>"
    )


def _metric_charts(scenarios: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for scenario in scenarios:
        summary = scenario.get("metric_summary", {})
        if not isinstance(summary, dict):
            continue
        summary = _display_metric_summary(
            summary,
            metadata=_metric_summary_metadata(scenario),
        )
        charts = [
            _chart_for_kind(
                summary,
                title="Command wall time: median",
                kind="wall",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="Timing: median",
                kind="timing",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="Memory: median",
                kind="memory",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="Throughput: median",
                kind="throughput",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="Quality scores: median",
                kind="quality_score",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="Quality error: median (lower is better)",
                kind="quality_error",
                primary_stat="median",
            ),
            _chart_for_kind(
                summary,
                title="PAI-Bench scores: median",
                kind="pai_bench_score",
                primary_stat="median",
            ),
        ]
        charts = [chart for chart in charts if chart]
        if not charts:
            continue
        scenario_id = html.escape(str(scenario.get("id", "")))
        sections.append(
            f"<section><h3>{scenario_id}</h3>"
            f'<div class="chart-grid">{"".join(charts)}</div></section>'
        )
    return "\n".join(sections)


def _chart_for_kind(
    summary: dict[str, Any],
    *,
    title: str,
    kind: str,
    primary_stat: str,
) -> str:
    items: list[tuple[str, dict[str, Any], MetricDisplay, float]] = []
    sorted_metrics = sorted(
        summary.items(),
        key=lambda item: _metric_sort_key(str(item[0])),
    )
    for metric, stats in sorted_metrics:
        if not isinstance(stats, dict):
            continue
        display = _metric_display(str(metric))
        if display.chart_kind != kind:
            continue
        primary = _scaled_stat(stats, primary_stat, display)
        if primary is None:
            continue
        items.append((str(metric), stats, display, primary))
    if not items:
        return ""

    if kind == "quality_score":
        scale_max = 1.0
    elif kind == "pai_bench_score":
        scale_max = 100.0
    else:
        scale_max = max(primary for _, _, _, primary in items)
    if scale_max <= 0:
        scale_max = 1.0

    rows = "\n".join(
        _chart_row(
            metric,
            display=display,
            primary=primary,
            scale_max=scale_max,
        )
        for metric, _, display, primary in items
    )
    return f"""
      <div class="chart-card">
        <h4>{html.escape(title)}</h4>
        <table>
          <thead>
            <tr><th>Metric</th><th>Median</th></tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    """


def _chart_row(
    metric: str,
    *,
    display: MetricDisplay,
    primary: float,
    scale_max: float,
) -> str:
    primary_pct = _bar_percent(primary, scale_max)
    return (
        "<tr>"
        f"<td>{html.escape(display.label)}<br><code>{html.escape(metric)}</code></td>"
        f"<td>{_value_bar(primary, primary_pct, display=display)}</td>"
        "</tr>"
    )


def _value_bar(
    value: float,
    percent: float,
    *,
    display: MetricDisplay,
) -> str:
    return (
        '<div class="value-bar">'
        '<div class="bar-track">'
        f'<span class="bar-fill" style="width: {percent:.1f}%"></span>'
        "</div>"
        f'<span class="chart-value">{_format_scaled_number(value, display.unit)}'
        f"{' ' + html.escape(display.unit) if display.unit else ''}</span>"
        "</div>"
    )


def _scaled_stat(
    stats: dict[str, Any], stat_name: str, display: MetricDisplay
) -> float | None:
    value = stats.get(stat_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) * display.scale


def _bar_percent(value: float, scale_max: float) -> float:
    if scale_max <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * value / scale_max))


def _quality_payload(
    result: dict[str, Any],
    *,
    manifest_output_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    metrics_path = result.get("metrics_path")
    if not isinstance(metrics_path, str):
        return {}
    path = _artifact_path(
        metrics_path,
        manifest_output_root=manifest_output_root,
        asset_root=asset_root,
    )
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_href(
    path_value: str,
    *,
    manifest_output_root: Path,
    asset_root: Path,
    page_dir: Path,
) -> str:
    path = _artifact_path(
        path_value,
        manifest_output_root=manifest_output_root,
        asset_root=asset_root,
    )
    return _file_href(path, page_dir=page_dir)


def _artifact_path(
    path_value: str,
    *,
    manifest_output_root: Path,
    asset_root: Path,
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return asset_root / path.relative_to(manifest_output_root)
        except ValueError:
            if asset_root != manifest_output_root:
                rel_to_manifest = os.path.relpath(path, start=manifest_output_root)
                return asset_root / rel_to_manifest
            return path
    return asset_root / path


def _file_href(path: Path, *, page_dir: Path) -> str:
    return Path(os.path.relpath(path, start=page_dir)).as_posix()


def _summary_stat(
    summary: dict[str, Any],
    metric: str,
    statistic: str,
) -> float | int | None:
    stats = summary.get(metric)
    if not isinstance(stats, dict):
        return None
    value = stats.get(statistic)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _display_metric_summary(
    summary: dict[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Fold parsed log-summary metrics into base metrics for HTML presentation."""
    base_metrics = {
        str(metric)
        for metric, stats in summary.items()
        if isinstance(stats, dict)
        and _foldable_derived_summary_metric(str(metric), metadata) is None
    }
    display: dict[str, dict[str, Any]] = {
        str(metric): dict(stats)
        for metric, stats in summary.items()
        if isinstance(stats, dict) and str(metric) in base_metrics
    }
    derived_folded_metrics: set[str] = set()

    for metric, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        metric = str(metric)
        derived = _foldable_derived_summary_metric(metric, metadata)
        if derived is None:
            if metric not in display:
                display[metric] = dict(stats)
            continue
        base_metric, statistic = derived
        if base_metric in display and base_metric not in derived_folded_metrics:
            continue
        value = stats.get("median")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        folded = display.get(base_metric)
        if folded is None:
            folded = {}
            display[base_metric] = folded
            derived_folded_metrics.add(base_metric)
        folded[statistic] = value
        count = stats.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            folded["count"] = max(int(folded.get("count", 0)), count)
    return display


def _metric_summary_metadata(scenario: dict[str, Any]) -> Mapping[str, Any]:
    metadata = scenario.get("metric_summary_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _foldable_derived_summary_metric(
    metric: str,
    metadata: Mapping[str, Any],
) -> tuple[str, str] | None:
    derived = _derived_summary_metric(metric)
    if derived is None:
        return None
    metric_metadata = metadata.get(metric)
    if not isinstance(metric_metadata, dict):
        return None
    record_types = metric_metadata.get("record_types")
    if not isinstance(record_types, list):
        return None
    normalized_record_types = {str(record_type) for record_type in record_types}
    if normalized_record_types != {"log_summary"}:
        return None
    parsers = metric_metadata.get("parsers")
    if not isinstance(parsers, list):
        return None
    normalized_parsers = {str(parser) for parser in parsers}
    if normalized_parsers != {"perf_summary"}:
        return None
    return derived


def _derived_summary_metric(metric: str) -> tuple[str, str] | None:
    for suffix, statistic in _DERIVED_SUMMARY_SUFFIXES:
        if metric.endswith(suffix):
            base_metric = metric[: -len(suffix)]
            if base_metric:
                return base_metric, statistic
    return None


def _numeric_or_none(
    value: object,
    *,
    fallback: object = None,
) -> float | int | None:
    if isinstance(value, bool):
        return _numeric_or_none(fallback)
    if isinstance(value, (int, float)):
        return value
    if isinstance(fallback, bool):
        return None
    return fallback if isinstance(fallback, (int, float)) else None


def _float_or_default(value: object, default: float) -> float:
    numeric = _numeric_or_none(value, fallback=default)
    if numeric is None:
        return default
    return float(numeric)


def _metric_hint(metric: str) -> str:
    hints = {
        "command_wall_s": "Full scenario process time: startup, model work, quality hooks, and file writing.",
        "quality_score": "Overall 0-1 summary; higher is better.",
        "quality_similarity_score": "0-1 closeness to the baseline MP4; higher is better.",
        "quality_visual_sanity_score": "No-reference check for blank, flat, striped, or unstable output; higher is better.",
        "quality_temporal_score": "Frame-to-frame stability proxy; higher is better.",
        "quality_ssim_score": "Structural similarity to baseline on sampled frames; higher is better.",
        "quality_rmse": "8-bit pixel RMSE against baseline; lower is better and 0 is identical.",
        "quality_mean_abs": "8-bit mean absolute pixel difference against baseline; lower is better and 0 is identical.",
        "quality_psnr_db": "Pixel similarity in dB; higher is better.",
        "pai_bench_g_score": "0-100 aggregate PAI-Bench-G evaluator score; higher is better.",
        "pai_bench_long_score": "0-100 aggregate long-video PAI-Bench evaluator score; higher is better.",
    }
    hint = hints.get(metric)
    if hint is None:
        return ""
    return f'<span class="metric-help">{html.escape(hint)}</span>'


def _metric_judgement_html(metric: str, value: object) -> str:
    judgement = _metric_judgement(metric, value)
    if judgement is None:
        return ""
    label, css_class = judgement
    return f'<span class="metric-pill {css_class}">{html.escape(label)}</span>'


def _metric_judgement(metric: str, value: object) -> tuple[str, str] | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if metric in {
        "quality_score",
        "quality_similarity_score",
        "quality_ssim_score",
        "quality_visual_sanity_score",
        "quality_temporal_score",
    }:
        if numeric >= 0.90:
            return "good", "good"
        if numeric >= 0.75:
            return "watch", "warn"
        return "investigate", "bad"
    if metric == "quality_rmse":
        if numeric <= 15.0:
            return "small", "good"
        if numeric <= 40.0:
            return "visible", "warn"
        return "large", "bad"
    if metric == "quality_mean_abs":
        if numeric <= 8.0:
            return "small", "good"
        if numeric <= 20.0:
            return "visible", "warn"
        return "large", "bad"
    if metric == "quality_psnr_db":
        if numeric >= 30.0:
            return "good", "good"
        if numeric >= 20.0:
            return "visible", "warn"
        return "large drift", "bad"
    if metric.startswith("pai_bench_") and metric.endswith("_score"):
        if numeric >= 75.0:
            return "good", "good"
        if numeric >= 50.0:
            return "watch", "warn"
        return "investigate", "bad"
    return None


def _gpu_summary(environment: dict[str, Any]) -> str:
    torch_devices = _dig(environment, "torch", "devices")
    if isinstance(torch_devices, list) and torch_devices:
        return "<br>".join(
            html.escape(
                f"{device.get('index')}: {device.get('name')} "
                f"({_float_or_default(device.get('total_memory_gib'), 0.0):.1f} GiB)"
            )
            for device in torch_devices
            if isinstance(device, dict)
        )
    nvidia_smi = environment.get("nvidia_smi")
    if isinstance(nvidia_smi, list) and nvidia_smi:
        return "<br>".join(
            html.escape(
                f"{gpu.get('index')}: {gpu.get('name')} "
                f"({gpu.get('memory_total_mib')} MiB, driver {gpu.get('driver_version')})"
            )
            for gpu in nvidia_smi
            if isinstance(gpu, dict)
        )
    return '<span class="muted">not detected</span>'


def _quality_baseline_summary(value: object) -> str:
    if not isinstance(value, dict):
        return '<span class="muted">not enabled</span>'
    baseline_dir = value.get("baseline_dir")
    if not baseline_dir:
        return '<span class="muted">not enabled</span>'
    details = [
        f"region={value.get('compare_region', '')}",
        f"samples={value.get('sample_count', '')}",
    ]
    frame_indices = value.get("frame_indices")
    if frame_indices:
        details.append(f"frames={frame_indices}")
    if value.get("compute_flip"):
        details.append("flip=true")
    return (
        f"<code>{_escape(baseline_dir)}</code><br>"
        f'<span class="muted">{_escape(", ".join(details))}</span>'
    )


def _metric_display(metric: str) -> MetricDisplay:
    label = _LABEL_OVERRIDES.get(metric)
    base_metric = metric
    statistic: str | None = None
    unit = ""
    scale = 1.0
    chart_kind = "other"

    for suffix, suffix_unit, suffix_scale, suffix_kind in (
        ("_median_s", "ms", 1000.0, "timing"),
        ("_p90_s", "ms", 1000.0, "timing"),
        ("_mean_s", "ms", 1000.0, "timing"),
        ("_min_s", "ms", 1000.0, "timing"),
        ("_max_s", "ms", 1000.0, "timing"),
        ("_median_fps", "fps", 1.0, "throughput"),
        ("_p90_fps", "fps", 1.0, "throughput"),
        ("_mean_fps", "fps", 1.0, "throughput"),
        ("_min_fps", "fps", 1.0, "throughput"),
        ("_max_fps", "fps", 1.0, "throughput"),
    ):
        if metric.endswith(suffix):
            base_metric = metric[: -len(suffix)]
            statistic = suffix.split("_")[1]
            unit = suffix_unit
            scale = suffix_scale
            chart_kind = suffix_kind
            break

    if not unit:
        if metric == "command_wall_s":
            unit = "s"
            scale = 1.0
            chart_kind = "wall"
        elif metric.endswith("_s"):
            base_metric = metric[:-2]
            unit = "ms"
            scale = 1000.0
            chart_kind = "timing"
        elif metric.endswith("_fps"):
            base_metric = metric[:-4]
            unit = "fps"
            chart_kind = "throughput"
        elif metric.endswith("_gib"):
            base_metric = metric[:-4]
            unit = "GiB"
            chart_kind = "memory"
        elif metric.endswith("_mib"):
            base_metric = metric[:-4]
            unit = "MiB"
            chart_kind = "memory"
        elif metric.startswith("quality_"):
            unit, chart_kind = _quality_unit_and_chart(metric)
        elif metric.startswith("pai_bench_"):
            unit, chart_kind = _pai_bench_unit_and_chart(metric)

    if base_metric.endswith("_s") and unit == "ms":
        base_metric = base_metric[:-2]
    if base_metric.endswith("_fps") and unit == "fps":
        base_metric = base_metric[:-4]

    if label is None:
        if metric.startswith("pai_bench_"):
            label = _pai_bench_label(metric)
        else:
            label = _LABEL_OVERRIDES.get(base_metric) or _title_metric(base_metric)
        if statistic is not None:
            label = f"{label} {statistic.upper() if statistic == 'p90' else statistic}"
    return MetricDisplay(label=label, unit=unit, scale=scale, chart_kind=chart_kind)


def _metric_sort_key(metric: str) -> tuple[int, str]:
    display = _metric_display(metric)
    base = metric
    for suffix in (
        "_median_s",
        "_p90_s",
        "_mean_s",
        "_min_s",
        "_max_s",
        "_median_fps",
        "_p90_fps",
        "_mean_fps",
        "_min_fps",
        "_max_fps",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return (_CHART_ORDER.get(metric, _CHART_ORDER.get(base, 100)), display.label)


def _title_metric(metric: str) -> str:
    words = metric.split("_")
    return " ".join(_title_word(word) for word in words if word)


def _title_word(word: str) -> str:
    upper_words = {"cpu", "cuda", "fps", "gpu", "rgb", "vae"}
    if word in upper_words:
        return word.upper()
    return word.capitalize()


def _quality_unit_and_chart(metric: str) -> tuple[str, str]:
    if metric.endswith("_score"):
        return "score", "quality_score"
    if metric.endswith("_db"):
        return "dB", "other"
    if metric.endswith("_count"):
        return "frames", "other"
    if "flip" in metric:
        return "score", "other"
    if "rmse" in metric or "mean_abs" in metric:
        return "px", "quality_error"
    return "", "other"


def _pai_bench_unit_and_chart(metric: str) -> tuple[str, str]:
    if metric.endswith("_score"):
        return "score", "pai_bench_score"
    if metric.endswith("_evaluated"):
        return "count", "other"
    return "", "other"


def _pai_bench_label(metric: str) -> str:
    if metric.startswith("pai_bench_long_"):
        suffix = metric.removeprefix("pai_bench_long_")
        prefix = "PAI-Bench-Long"
    elif metric.startswith("pai_bench_g_"):
        suffix = metric.removeprefix("pai_bench_g_")
        prefix = "PAI-Bench-G"
    else:
        return _title_metric(metric)
    suffix = suffix.removesuffix("_score")
    suffix = suffix.removesuffix("_evaluated")
    if not suffix:
        return f"{prefix} score"
    return f"{prefix} {_title_metric(suffix)}"


def _format_wall_time(value: object) -> str:
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 1.0:
            return f"{seconds * 1000.0:.1f} ms"
        if seconds < 60.0:
            return f"{seconds:.2f} s"
        minutes, remainder = divmod(seconds, 60.0)
        if minutes < 60:
            return f"{int(minutes)}m {remainder:.1f}s"
        hours, minutes = divmod(int(minutes), 60)
        return f"{hours}h {minutes:02d}m {remainder:.1f}s"
    return ""


def _format_metric_value(value: object, display: MetricDisplay) -> str:
    if isinstance(value, (int, float)):
        return _format_scaled_number(float(value) * display.scale, display.unit)
    return ""


def _format_metric_with_unit(value: float | int, display: MetricDisplay) -> str:
    formatted = _format_metric_value(value, display)
    if not formatted:
        return ""
    if display.unit in {
        "ms",
        "s",
        "GiB",
        "MiB",
        "fps",
        "dB",
        "frames",
        "px",
        "score",
        "count",
    }:
        return f"{formatted} {display.unit}"
    return formatted


def _format_scaled_number(value: float, unit: str) -> str:
    numeric = float(value)
    if not unit and numeric.is_integer():
        return str(int(numeric))
    magnitude = abs(numeric)
    if unit == "ms":
        if magnitude >= 100:
            return f"{numeric:.1f}"
        if magnitude >= 10:
            return f"{numeric:.2f}"
        if magnitude >= 1:
            return f"{numeric:.3f}"
        return f"{numeric:.4f}"
    if unit == "count":
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:.3f}"
    if unit in {"s", "GiB", "MiB", "fps", "score", "dB", "frames", "px"}:
        if magnitude >= 100:
            return f"{numeric:.1f}"
        if magnitude >= 10:
            return f"{numeric:.2f}"
        if magnitude >= 1:
            return f"{numeric:.3f}"
        return f"{numeric:.4f}"
    return f"{numeric:.6g}"


def _first_line(value: object) -> str:
    lines = str(value or "").splitlines()
    return lines[0] if lines else ""


def _empty_charts() -> str:
    return (
        '<p class="muted">No wall-time, timing, memory, throughput, or '
        "quality metrics were available for charting.</p>"
    )


def _empty_highlights() -> str:
    return '<p class="muted">No scenario highlight metrics were available.</p>'


def _empty_model_reports() -> str:
    return '<p class="muted">No model detail reports were generated.</p>'


def _empty_quality_comparisons() -> str:
    return (
        '<p class="muted">No baseline quality comparisons were available. '
        "Run with <code>--quality-baseline-dir</code> to compare MP4s.</p>"
    )


def _empty_metrics_row() -> str:
    return (
        '<tr><td colspan="5" class="muted">No numeric metrics were collected.</td></tr>'
    )


def _escape(value: object) -> str:
    return html.escape(str(value))


def _dig(data: object, *keys: str) -> object:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
