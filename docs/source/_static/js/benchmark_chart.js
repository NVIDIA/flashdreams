// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Generic benchmark chart renderer.
// Required container attributes:
// - data-benchmark-md-url: markdown table path (served under docs static)
// - data-benchmark-series: "key:Label:#RRGGBB;key2:Label2:#RRGGBB"

function addSvgNode(parent, tag, attrs = {}, text = "") {
  const ns = "http://www.w3.org/2000/svg";
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
  if (text) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function parseSeriesSpec(seriesSpec) {
  const entries = (seriesSpec || "")
    .split(";")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) {
    throw new Error("data-benchmark-series is empty.");
  }

  return entries.map((entry) => {
    const parts = entry.split(":");
    if (parts.length !== 3) {
      throw new Error(
        `Invalid series entry "${entry}". Expected "key:Label:#RRGGBB".`,
      );
    }
    return {
      key: parts[0].trim().toLowerCase(),
      label: parts[1].trim(),
      color: parts[2].trim(),
    };
  });
}

function parseBenchmarkMarkdown(markdownText, series) {
  const lines = markdownText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const tableLines = lines.filter((line) => line.startsWith("|"));
  if (tableLines.length < 3) {
    throw new Error("Benchmark markdown table is missing or malformed.");
  }

  const parseRow = (line) =>
    line
      .split("|")
      .map((cell) => cell.trim())
      .filter((cell) => cell.length > 0);

  const header = parseRow(tableLines[0]).map((h) => h.toLowerCase());
  if (header.length === 0) {
    throw new Error("Benchmark markdown header is empty.");
  }
  const groupIdx = 0;

  const seriesIndices = series.map((s) => {
    const idx = header.indexOf(s.key);
    if (idx < 0) {
      throw new Error(`Header missing required series column "${s.key}".`);
    }
    return idx;
  });

  return tableLines.slice(2).map((line) => {
    const row = parseRow(line);
    const parsed = { device: row[groupIdx] };
    series.forEach((s, i) => {
      parsed[s.key] = Number(row[seriesIndices[i]]);
    });
    return parsed;
  });
}

async function loadBenchmarkData(mdUrl, series) {
  const response = await fetch(mdUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load benchmark markdown: ${response.status}`);
  }
  return parseBenchmarkMarkdown(await response.text(), series);
}

function renderBenchmarkChart(container, benchmarkData, series) {
  if (!container || !benchmarkData || benchmarkData.length === 0) return;

  const width = 864;
  const height = 490;
  const left = 60;
  const right = 60;
  const top = 56;
  const bottom = 52;
  const chartW = width - left - right;
  const chartH = height - top - bottom;
  const groupGap = 48;
  const groupW = (chartW - groupGap * (benchmarkData.length - 1)) / benchmarkData.length;
  const barGap = 14;
  const barWidthScale = 0.78;
  const fullBarW = (groupW - barGap * (series.length - 1)) / series.length;
  const barW = fullBarW * barWidthScale;
  const groupInnerW = barW * series.length + barGap * (series.length - 1);
  const groupInnerOffset = (groupW - groupInnerW) / 2;

  const maxMs = Math.max(
    ...benchmarkData.flatMap((d) => series.map((s) => Number(d[s.key] || 0))),
  );
  if (maxMs <= 0) return;

  container.textContent = "";
  const ariaLabel = container.dataset.chartAriaLabel || "Benchmark chart";

  const svg = addSvgNode(container, "svg", {
    class: "benchmark-chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    "aria-label": ariaLabel,
  });
  addSvgNode(svg, "rect", {
    x: 0,
    y: 0,
    width,
    height,
    fill: "#f8fafc",
  });
  addSvgNode(svg, "line", {
    x1: left,
    y1: top + chartH,
    x2: width - right,
    y2: top + chartH,
    stroke: "#cbd5e1",
    "stroke-width": 1.5,
  });

  const legendMeasureCtx = document.createElement("canvas").getContext("2d");
  if (legendMeasureCtx) {
    legendMeasureCtx.font = "700 20px DejaVu Sans, sans-serif";
  }
  const legendY = 30;
  const legendRectW = 20;
  const legendRectLabelGap = 8;
  const legendItemGap = 36;
  const legendTextWidths = series.map((s) =>
    legendMeasureCtx ? legendMeasureCtx.measureText(s.label).width : s.label.length * 10,
  );
  const legendItemWidths = legendTextWidths.map(
    (w) => legendRectW + legendRectLabelGap + w,
  );
  const legendTotalW =
    legendItemWidths.reduce((acc, w) => acc + w, 0) + legendItemGap * (series.length - 1);
  let legendX = (width - legendTotalW) / 2;
  series.forEach((s) => {
    addSvgNode(svg, "rect", {
      x: legendX,
      y: legendY - 12,
      width: 20,
      height: 12,
      fill: s.color,
    });
    addSvgNode(
      svg,
      "text",
      {
        x: legendX + 28,
        y: legendY - 2,
        "font-size": 20,
        "font-weight": 700,
        fill: "#111827",
        "text-anchor": "start",
        "font-family": "inherit",
      },
      s.label,
    );
    legendX +=
      legendRectW +
      legendRectLabelGap +
      legendTextWidths[series.indexOf(s)] +
      legendItemGap;
  });

  let x = left;
  benchmarkData.forEach((row) => {
    series.forEach((s, idx) => {
      const value = Number(row[s.key]);
      const barH = (value / maxMs) * (chartH * 0.88);
      const barX = x + groupInnerOffset + idx * (barW + barGap);
      const barY = top + chartH - barH;
      addSvgNode(svg, "rect", {
        x: barX.toFixed(2),
        y: barY.toFixed(2),
        width: barW.toFixed(2),
        height: barH.toFixed(2),
        fill: s.color,
      });
      addSvgNode(
        svg,
        "text",
        {
          x: (barX + barW / 2).toFixed(2),
          y: (barY - 8).toFixed(2),
          "font-size": 18,
          "font-weight": 700,
          fill: "#111827",
          "text-anchor": "middle",
          "font-family": "inherit",
        },
        String(Math.round(value)),
      );
    });

    addSvgNode(
      svg,
      "text",
      {
        x: (x + groupW / 2).toFixed(2),
        y: (top + chartH + 28).toFixed(2),
        "font-size": 24,
        "font-weight": 700,
        fill: "#111827",
        "text-anchor": "middle",
        "font-family": "inherit",
      },
      row.device,
    );
    x += groupW + groupGap;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-benchmark-md-url][data-benchmark-series]").forEach(
    (container) => {
      const mdUrl = container.dataset.benchmarkMdUrl;
      const seriesSpec = container.dataset.benchmarkSeries;
      try {
        const series = parseSeriesSpec(seriesSpec);
        loadBenchmarkData(mdUrl, series)
          .then((data) => renderBenchmarkChart(container, data, series))
          .catch((error) => {
            container.textContent = `Failed to load benchmark chart data: ${error.message}`;
          });
      } catch (error) {
        container.textContent = `Failed to parse chart configuration: ${error.message}`;
      }
    },
  );
});
