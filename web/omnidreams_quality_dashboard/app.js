const DEFAULT_SUMMARY_PATH = "/outputs/omnidreams-quality-sweep/metrics_summary.json";
const DEFAULT_VLM_SUMMARY_NAME = "vlm_artifacts_summary.json";
const METRIC_DEFS = {
  niqe: { label: "NIQE", better: "lower", colorClass: "niqe" },
  musiq: { label: "MUSIQ", better: "higher", colorClass: "musiq" },
  clipiqa: { label: "CLIPIQA", better: "higher", colorClass: "clipiqa" },
};
const ARTIFACT_DEFS = {
  hallucinated_vehicle: { label: "Hallucinated vehicle" },
  sign_glyph: { label: "Sign glyph" },
  traffic_light: { label: "Traffic light" },
  lane_geometry: { label: "Lane geometry" },
  road_user_anomaly: { label: "Road user anomaly" },
  temporal_inconsistency: { label: "Temporal inconsistency" },
};

const state = {
  summary: null,
  vlmSummary: null,
  vlmMap: new Map(),
  rawItems: [],
  items: [],
  ranges: {},
  sortMetric: "composite",
  sortOrder: "best",
  viewMode: "all",
  videoMode: "generated",
  artifactFilter: "all",
  query: "",
  selectedId: null,
};

const els = {
  summaryLine: document.getElementById("summaryLine"),
  summaryMetrics: document.getElementById("summaryMetrics"),
  sortMetric: document.getElementById("sortMetric"),
  sortOrder: document.getElementById("sortOrder"),
  viewMode: document.getElementById("viewMode"),
  videoMode: document.getElementById("videoMode"),
  artifactFilter: document.getElementById("artifactFilter"),
  searchBox: document.getElementById("searchBox"),
  selectedTitle: document.getElementById("selectedTitle"),
  selectedRank: document.getElementById("selectedRank"),
  mainVideo: document.getElementById("mainVideo"),
  selectedMeta: document.getElementById("selectedMeta"),
  selectedMetrics: document.getElementById("selectedMetrics"),
  selectedArtifacts: document.getElementById("selectedArtifacts"),
  topClips: document.getElementById("topClips"),
  resultCount: document.getElementById("resultCount"),
  rankTable: document.getElementById("rankTable"),
  resetFilters: document.getElementById("resetFilters"),
  errorBox: document.getElementById("errorBox"),
};

function getSummaryPath() {
  const params = new URLSearchParams(window.location.search);
  return params.get("summary") || DEFAULT_SUMMARY_PATH;
}

function getVlmSummaryPath() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("vlm")) return params.get("vlm");
  const summaryPath = getSummaryPath();
  return `${dirname(summaryPath)}/${DEFAULT_VLM_SUMMARY_NAME}`;
}

function dirname(path) {
  return path.slice(0, path.lastIndexOf("/"));
}

function toUrl(path, summary) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("/")) {
    const summaryPath = getSummaryPath();
    const summaryBase = dirname(summary?.sourcePath || summaryPath);
    const root = summary?.root || "";
    if (root && path.startsWith(root)) {
      return encodeURI(summaryBase + path.slice(root.length));
    }
    const outputIndex = path.indexOf("/outputs/");
    if (outputIndex >= 0) {
      return encodeURI(path.slice(outputIndex));
    }
    return encodeURI(path);
  }
  return encodeURI(path);
}

function relativeOutputDir(record, detail, summary) {
  if (detail?.relative_output_dir) return detail.relative_output_dir;
  const root = summary?.root || "";
  const source = record.source_video || "";
  if (root && source.startsWith(root)) {
    return dirname(source.slice(root.length).replace(/^\//, ""));
  }
  const parts = source.split("/outputs/");
  if (parts.length > 1) {
    return dirname(parts[1]).replace(/^omnidreams-quality-sweep\//, "");
  }
  return dirname(source);
}

function parseOutput(rel) {
  const parts = rel.split("/").filter(Boolean);
  const sampleIndex = parts.indexOf("omni-dreams-samples");
  const clipId = sampleIndex >= 0 ? parts[sampleIndex + 1] : parts.at(-4) || "unknown";
  const recipe = sampleIndex >= 0 ? parts[sampleIndex + 2] : parts.at(-3) || "unknown";
  const seed = parts.at(-1) || "unknown";
  const stream = parts.at(-2) || "unknown";
  return { clipId, recipe, seed, stream };
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function formatScore(value) {
  if (!Number.isFinite(value)) return "-";
  return Math.round(value * 1000) / 10;
}

function formatArtifactName(name) {
  if (!name) return "-";
  return ARTIFACT_DEFS[name]?.label || name.replaceAll("_", " ");
}

function severityLabel(value) {
  if (!Number.isFinite(value)) return "-";
  return ["None", "Minor", "Review", "Severe"][value] || String(value);
}

function severityClass(value) {
  if (!Number.isFinite(value)) return "severity-unknown";
  return `severity-${Math.max(0, Math.min(3, value))}`;
}

function artifactStatusLabel(item) {
  if (!item.artifacts?.available) return "No VLM";
  if (item.artifacts.responseValid === false) return "Schema issue";
  return severityLabel(item.artifacts.overall);
}

function normalize(value, metric) {
  const range = state.ranges[metric];
  if (!range || !Number.isFinite(value)) return 0;
  const span = range.max - range.min;
  if (span <= 0) return 1;
  if (METRIC_DEFS[metric].better === "lower") {
    return (range.max - value) / span;
  }
  return (value - range.min) / span;
}

function computeRanges(items) {
  const ranges = {};
  for (const metric of Object.keys(METRIC_DEFS)) {
    const values = items
      .map((item) => item.metrics[metric])
      .filter((value) => Number.isFinite(value));
    ranges[metric] = {
      min: Math.min(...values),
      max: Math.max(...values),
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    };
  }
  state.ranges = ranges;
}

function addCompositeScores(items) {
  for (const item of items) {
    item.normalized = {};
    for (const metric of Object.keys(METRIC_DEFS)) {
      item.normalized[metric] = normalize(item.metrics[metric], metric);
    }
    const values = Object.values(item.normalized);
    item.composite = values.reduce((sum, value) => sum + value, 0) / values.length;
  }
}

function metricValue(item, metric) {
  if (metric === "composite") return item.composite;
  if (metric === "artifact_severity") return item.artifacts?.overall ?? -1;
  if (ARTIFACT_DEFS[metric]) {
    return item.artifacts?.scores?.[metric]?.severity ?? -1;
  }
  return item.metrics[metric];
}

function compareItems(a, b, metric = state.sortMetric, order = state.sortOrder) {
  const aValue = metricValue(a, metric);
  const bValue = metricValue(b, metric);
  const higherIsBetter =
    metric === "composite" ||
    metric === "artifact_severity" ||
    Boolean(ARTIFACT_DEFS[metric]) ||
    METRIC_DEFS[metric]?.better === "higher";
  let delta = higherIsBetter ? bValue - aValue : aValue - bValue;
  if (order === "worst") delta = -delta;
  if (delta !== 0) return delta;
  return a.outputRel.localeCompare(b.outputRel);
}

function bestItem(items, metric = state.sortMetric) {
  return [...items].sort((a, b) => compareItems(a, b, metric, "best"))[0];
}

function filteredItems() {
  const query = state.query.trim().toLowerCase();
  let items = state.rawItems;
  if (query) {
    items = items.filter((item) => item.searchText.includes(query));
  }
  if (state.artifactFilter === "needs-review") {
    items = items.filter((item) => item.artifacts?.needsReview);
  } else if (state.artifactFilter === "schema-warning") {
    items = items.filter((item) => item.artifacts?.responseValid === false);
  } else if (state.artifactFilter === "clean") {
    items = items.filter(
      (item) => item.artifacts?.available && item.artifacts.responseValid !== false && item.artifacts.overall === 0
    );
  } else if (ARTIFACT_DEFS[state.artifactFilter]) {
    items = items.filter(
      (item) => (item.artifacts?.scores?.[state.artifactFilter]?.severity ?? 0) > 0
    );
  }
  if (state.viewMode === "best-per-clip") {
    const groups = new Map();
    for (const item of items) {
      if (!groups.has(item.clipId)) groups.set(item.clipId, []);
      groups.get(item.clipId).push(item);
    }
    items = [...groups.values()].map((group) => bestItem(group));
  }
  return [...items].sort(compareItems);
}

function aggregate(items) {
  const out = {};
  for (const metric of Object.keys(METRIC_DEFS)) {
    const values = items.map((item) => item.metrics[metric]).filter(Number.isFinite);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    out[metric] = { mean, count: values.length };
  }
  return out;
}

function aggregateArtifacts(items) {
  const withVlm = items.filter((item) => item.artifacts?.available);
  const review = withVlm.filter((item) => item.artifacts.needsReview);
  const invalid = withVlm.filter((item) => item.artifacts.responseValid === false);
  const maxSeverity = withVlm.reduce(
    (maxValue, item) => Math.max(maxValue, item.artifacts.overall ?? 0),
    0
  );
  return {
    withVlm: withVlm.length,
    review: review.length,
    invalid: invalid.length,
    maxSeverity,
  };
}

function metaCell(label, value) {
  return `<div class="meta-cell"><span>${label}</span><strong title="${value}">${value}</strong></div>`;
}

function metricRow(metric, item) {
  const value = item.metrics[metric];
  const pct = Math.max(0, Math.min(100, item.normalized[metric] * 100));
  const def = METRIC_DEFS[metric];
  return `
    <div class="metric-row">
      <div class="metric-label">${def.label}</div>
      <div class="metric-track" aria-label="${def.label}">
        <div class="metric-fill ${def.colorClass}" style="width: ${pct}%"></div>
      </div>
      <div class="metric-value">${formatNumber(value, metric === "clipiqa" ? 4 : 3)}</div>
    </div>
  `;
}

function artifactBadge(item) {
  const overall = item.artifacts?.overall;
  if (!item.artifacts?.available) {
    return `<span class="artifact-badge severity-unknown">No VLM</span>`;
  }
  if (item.artifacts.responseValid === false) {
    return `<span class="artifact-badge severity-invalid">Schema issue</span>`;
  }
  return `<span class="artifact-badge ${severityClass(overall)}">${severityLabel(overall)}</span>`;
}

function renderArtifactPanel(item) {
  if (!item.artifacts?.available) {
    return `
      <div class="artifact-empty">
        VLM artifact results were not found for this output.
      </div>
    `;
  }

  const topCategories =
    item.artifacts.highestCategories?.length > 0
      ? item.artifacts.highestCategories.map(formatArtifactName).join(", ")
      : "None";
  const sheetLink = item.artifacts.contactSheetUrl
    ? `<a href="${item.artifacts.contactSheetUrl}" target="_blank" rel="noreferrer">Open contact sheet</a>`
    : "";
  const schemaWarning =
    item.artifacts.responseValid === false
      ? `
        <div class="artifact-warning">
          <strong>Schema warning</strong>
          <span>${item.artifacts.parseWarnings?.join(" · ") || "The VLM response did not match the expected schema."}</span>
        </div>
      `
      : "";
  const categoryRows = Object.entries(ARTIFACT_DEFS)
    .map(([name, def]) => {
      const score = item.artifacts.scores?.[name] || {};
      const severity = score.severity ?? 0;
      const confidence = score.confidence ?? 0;
      const evidence = score.evidence || "";
      return `
        <div class="artifact-row ${severityClass(severity)}">
          <div>
            <strong>${def.label}</strong>
            <span>${evidence || "No evidence noted."}</span>
          </div>
          <div class="artifact-score">
            <b>${severity}</b>
            <span>${formatNumber(confidence, 2)}</span>
          </div>
        </div>
      `;
    })
    .join("");

  return `
    <div class="artifact-summary">
      <div>
        <p class="eyebrow">VLM Artifact Review</p>
        <h3>${artifactStatusLabel(item)} · ${topCategories}</h3>
      </div>
      ${sheetLink}
    </div>
    ${schemaWarning}
    <div class="artifact-rows">${categoryRows}</div>
    ${
      item.artifacts.contactSheetUrl
        ? `<img class="contact-sheet-preview" src="${item.artifacts.contactSheetUrl}" alt="VLM contact sheet" loading="lazy" />`
        : ""
    }
  `;
}

function selectItem(id) {
  const item = state.items.find((candidate) => candidate.id === id) || state.items[0];
  if (!item) return;
  state.selectedId = item.id;
  const rank = state.items.findIndex((candidate) => candidate.id === item.id) + 1;
  const videoUrl = state.videoMode === "stacked" ? item.sourceUrl : item.croppedUrl;

  els.selectedTitle.textContent = item.clipId;
  els.selectedRank.textContent = `#${rank}`;
  if (els.mainVideo.getAttribute("src") !== videoUrl) {
    els.mainVideo.setAttribute("src", videoUrl);
    els.mainVideo.load();
  }

  const resolution = item.resolution || "-";
  els.selectedMeta.innerHTML = [
    metaCell("Seed", item.seed),
    metaCell("Score", `${formatScore(item.composite)} / 100`),
    metaCell("VLM", artifactStatusLabel(item)),
    metaCell("Frames", item.frames || "-"),
    metaCell("Resolution", resolution),
    metaCell("Stream", item.stream),
    metaCell("Recipe", item.recipe),
    metaCell("Output", item.outputRel),
    metaCell("Video", state.videoMode === "stacked" ? "Stacked source" : "Generated crop"),
  ].join("");

  els.selectedMetrics.innerHTML = Object.keys(METRIC_DEFS)
    .map((metric) => metricRow(metric, item))
    .join("");
  els.selectedArtifacts.innerHTML = renderArtifactPanel(item);

  document
    .querySelectorAll("[data-item-id]")
    .forEach((node) => node.classList.toggle("is-selected", node.dataset.itemId === item.id));
}

function renderSummary() {
  const total = state.rawItems.length;
  const clips = new Set(state.rawItems.map((item) => item.clipId)).size;
  const stats = aggregate(state.rawItems);
  const artifactStats = aggregateArtifacts(state.rawItems);
  const vlmText = state.vlmSummary
    ? ` VLM: ${artifactStats.withVlm} reviewed, ${artifactStats.review} need review, ${artifactStats.invalid} schema warnings.`
    : " VLM: not loaded.";
  els.summaryLine.textContent = `${total} evaluated rollouts across ${clips} clips. Status: ${state.summary.status}.${vlmText}`;
  els.summaryMetrics.innerHTML = [
    ["Rollouts", total],
    ["Clips", clips],
    ["Needs Review", artifactStats.review],
    ["Mean MUSIQ", formatNumber(stats.musiq.mean, 2)],
  ]
    .map(([label, value]) => `<div class="summary-tile"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderTopClips() {
  const top = state.items.slice(0, 8);
  els.resultCount.textContent = `${state.items.length} shown`;
  els.topClips.innerHTML = top
    .map(
      (item, index) => `
        <div class="clip-card ${item.id === state.selectedId ? "is-selected" : ""}" data-item-id="${item.id}" role="button" tabindex="0">
          <div class="thumb">
            <video src="${item.croppedUrl}" muted loop playsinline preload="metadata"></video>
          </div>
          <div class="clip-copy">
            <p class="clip-title" title="${item.clipId}">#${index + 1} ${item.clipId}</p>
            <p class="clip-subtitle">seed ${item.seed} · NIQE ${formatNumber(item.metrics.niqe, 2)} · VLM ${artifactStatusLabel(item)}</p>
            <span class="score-pill">${formatScore(item.composite)} quality</span>
            ${artifactBadge(item)}
          </div>
        </div>
      `
    )
    .join("");
}

function renderTable() {
  els.rankTable.innerHTML = state.items
    .map((item, index) => {
      const resolution = item.resolution || "-";
      const topArtifact =
        item.artifacts?.highestCategories?.map(formatArtifactName).join(", ") || "-";
      return `
        <tr data-item-id="${item.id}" class="${item.id === state.selectedId ? "is-selected" : ""}" tabindex="0">
          <td class="numeric">${index + 1}</td>
          <td class="path-cell" title="${item.clipId}">${item.clipId}</td>
          <td class="numeric">${item.seed}</td>
          <td class="numeric">${formatScore(item.composite)}</td>
          <td class="numeric">${formatNumber(item.metrics.niqe, 3)}</td>
          <td class="numeric">${formatNumber(item.metrics.musiq, 3)}</td>
          <td class="numeric">${formatNumber(item.metrics.clipiqa, 4)}</td>
          <td class="numeric">${artifactBadge(item)}</td>
          <td class="path-cell" title="${topArtifact}">${topArtifact}</td>
          <td class="numeric">${item.frames || "-"}</td>
          <td>${resolution}</td>
        </tr>
      `;
    })
    .join("");
}

function render() {
  state.items = filteredItems();
  renderSummary();
  renderTopClips();
  renderTable();
  const nextSelection =
    state.items.find((item) => item.id === state.selectedId)?.id || state.items[0]?.id;
  if (nextSelection) selectItem(nextSelection);
}

function showError(error) {
  els.errorBox.hidden = false;
  els.errorBox.textContent = error.message || String(error);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  return response.json();
}

async function fetchJsonOptional(url) {
  const response = await fetch(url);
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  return response.json();
}

function fallbackArtifacts() {
  return {
    available: false,
    overall: null,
    needsReview: false,
    highestCategories: [],
    scores: {},
    contactSheetUrl: "",
    responseValid: null,
    parseWarnings: [],
  };
}

function buildArtifacts(record, detail, summary) {
  if (!record && !detail) return fallbackArtifacts();
  const artifacts = detail?.artifacts || {};
  const scores = artifacts.artifact_scores || record?.artifact_scores || {};
  const normalizedScores = {};
  for (const name of Object.keys(ARTIFACT_DEFS)) {
    normalizedScores[name] = {
      severity: scores[name]?.severity ?? 0,
      confidence: scores[name]?.confidence ?? 0,
      evidence: scores[name]?.evidence || "",
    };
  }
  const overall =
    artifacts.overall_artifact_severity ?? record?.overall_artifact_severity ?? 0;
  const highestCategories =
    artifacts.highest_severity_categories || record?.highest_severity_categories || [];
  const contactSheet = detail?.contact_sheet || record?.contact_sheet || "";
  const responseValid = artifacts.response_valid ?? record?.response_valid ?? true;
  const parseWarnings = artifacts.parse_warnings || record?.parse_warnings || [];
  const inferredNeedsReview = overall >= 2 || responseValid === false;
  return {
    available: true,
    overall,
    needsReview: artifacts.needs_review ?? record?.needs_review ?? inferredNeedsReview,
    highestCategories,
    scores: normalizedScores,
    contactSheetUrl: toUrl(contactSheet, summary),
    responseValid,
    parseWarnings,
  };
}

function buildVlmMap(vlmSummary, detailResults = []) {
  const records = vlmSummary?.records || [];
  const map = new Map();
  records.forEach((record, index) => {
    const detail =
      detailResults[index]?.status === "fulfilled" ? detailResults[index].value : null;
    map.set(record.relative_output_dir, buildArtifacts(record, detail, vlmSummary));
  });
  return map;
}

function buildItem(record, detail, summary, artifacts = null) {
  const outputRel = relativeOutputDir(record, detail, summary);
  const parsed = parseOutput(outputRel);
  const result = detail?.result || {};
  const resolution =
    Array.isArray(result.pred_final_resolution) && result.pred_final_resolution.length >= 2
      ? `${result.pred_final_resolution[1]}x${result.pred_final_resolution[0]}`
      : "";
  const metrics = detail?.metrics || record.metrics || {};
  const sourceUrl = toUrl(detail?.source_video || record.source_video, summary);
  const croppedUrl = toUrl(detail?.cropped_video || record.cropped_video, summary);
  const id = outputRel;
  const searchText = [
    outputRel,
    parsed.clipId,
    parsed.recipe,
    parsed.seed,
    parsed.stream,
    artifacts?.highestCategories?.join(" ") || "",
  ]
    .join(" ")
    .toLowerCase();

  return {
    id,
    outputRel,
    metrics,
    sourceUrl,
    croppedUrl,
    frames: result.num_frames || "",
    resolution,
    artifacts: artifacts || fallbackArtifacts(),
    searchText,
    ...parsed,
  };
}

async function loadDetails(records, summary) {
  return Promise.allSettled(
    records.map((record) => fetchJson(toUrl(record.metrics_json, summary)))
  );
}

async function loadVlmDetails(vlmSummary) {
  const records = vlmSummary?.records || [];
  return Promise.allSettled(
    records.map((record) => fetchJson(toUrl(record.vlm_artifacts_json, vlmSummary)))
  );
}

function refreshItems(records, detailResults = [], vlmMap = state.vlmMap) {
  state.rawItems = records.map((record, index) => {
    const detail =
      detailResults[index]?.status === "fulfilled" ? detailResults[index].value : null;
    const outputRel = relativeOutputDir(record, detail, state.summary);
    return buildItem(record, detail, state.summary, vlmMap.get(outputRel));
  });
  computeRanges(state.rawItems);
  addCompositeScores(state.rawItems);
  render();
}

function bindEvents() {
  els.sortMetric.addEventListener("change", () => {
    state.sortMetric = els.sortMetric.value;
    render();
  });
  els.sortOrder.addEventListener("change", () => {
    state.sortOrder = els.sortOrder.value;
    render();
  });
  els.viewMode.addEventListener("change", () => {
    state.viewMode = els.viewMode.value;
    render();
  });
  els.videoMode.addEventListener("change", () => {
    state.videoMode = els.videoMode.value;
    selectItem(state.selectedId);
  });
  els.artifactFilter.addEventListener("change", () => {
    state.artifactFilter = els.artifactFilter.value;
    render();
  });
  els.searchBox.addEventListener("input", () => {
    state.query = els.searchBox.value;
    render();
  });
  els.resetFilters.addEventListener("click", () => {
    els.sortMetric.value = "composite";
    els.sortOrder.value = "best";
    els.viewMode.value = "all";
    els.videoMode.value = "generated";
    els.artifactFilter.value = "all";
    els.searchBox.value = "";
    Object.assign(state, {
      sortMetric: "composite",
      sortOrder: "best",
      viewMode: "all",
      videoMode: "generated",
      artifactFilter: "all",
      query: "",
    });
    render();
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-item-id]");
    if (target) selectItem(target.dataset.itemId);
    const sortTarget = event.target.closest("[data-sort]");
    if (sortTarget) {
      state.sortMetric = sortTarget.dataset.sort;
      els.sortMetric.value = state.sortMetric;
      render();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = event.target.closest("[data-item-id]");
    if (!target) return;
    event.preventDefault();
    selectItem(target.dataset.itemId);
  });
}

async function init() {
  bindEvents();
  try {
    const [summary, vlmSummary] = await Promise.all([
      fetchJson(getSummaryPath()),
      fetchJsonOptional(getVlmSummaryPath()),
    ]);
    summary.sourcePath = getSummaryPath();
    if (vlmSummary) vlmSummary.sourcePath = getVlmSummaryPath();
    state.summary = summary;
    state.vlmSummary = vlmSummary;
    state.vlmMap = buildVlmMap(vlmSummary);
    const records = (summary.records || []).filter((record) => record.status === "ok");
    refreshItems(records);
    const detailResults = await loadDetails(records, summary);
    refreshItems(records, detailResults);
    if (vlmSummary) {
      const vlmDetailResults = await loadVlmDetails(vlmSummary);
      state.vlmMap = buildVlmMap(vlmSummary, vlmDetailResults);
      refreshItems(records, detailResults, state.vlmMap);
    }
  } catch (error) {
    showError(error);
    els.summaryLine.textContent = "Unable to load metrics.";
  }
}

init();
