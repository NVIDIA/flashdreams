// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const peer = new RTCPeerConnection();
const controls = peer.createDataChannel("controls");
const pointerControls = peer.createDataChannel("pointer-controls");
peer.addTransceiver("video", {direction: "recvonly"});
const video = document.getElementById("video");
const status = document.getElementById("status");
const latencyPanel = document.getElementById("input-latency");
const latencyValue = document.getElementById("latencyValue");
const latencyState = document.getElementById("latencyState");
const latencyEventLabel = document.getElementById("latencyEventLabel");
const latencyEventTime = document.getElementById("latencyEventTime");
const latencyFrameTime = document.getElementById("latencyFrameTime");
const latencyP50 = document.getElementById("latencyP50");
const latencyP90 = document.getElementById("latencyP90");
const latencyDetail = document.getElementById("latencyDetail");
const pressedKeys = new Map();
const pressedButtons = new Set();
let lastPointerPosition = {x: 0, y: 0};

const UINT32_MODULUS = 0x100000000;
const MAX_PENDING_INPUTS = 4096;
const MAX_FRAME_CACHE = 512;
const MAX_COMPLETED_IDS = 8192;
const MAX_LATENCY_SAMPLES = 240;
const INPUT_TRACE_TTL_MS = 10 * 60 * 1000;
const FRAME_TRACE_TTL_MS = 30 * 1000;
const RTP_TIMESTAMP_TOLERANCE_TICKS = 2;
const MAX_NONCRITICAL_BUFFER_BYTES = 4 * 1024;

const pendingInputs = new Map();
const frameMarkers = new Map();
const renderedFrames = new Map();
const completedEventIds = new Map();
const completedFrameIds = new Map();
const renderedFrameSignatures = new Map();
const latencySamples = [];

let inputSequence = 0;
let renderedFrameSequence = 0;
let latestPanelRecord = null;
let panelRenderHandle = null;
let rtpTimestampOffset = null;
let rtpCalibrationConflicted = false;
let lastCachePruneAtMs = Number.NEGATIVE_INFINITY;
let pendingPointerMove = null;
let pointerMoveHandle = null;
let pendingWheel = null;
let wheelHandle = null;

const cancelPendingPointerMove = () => {
  if (pointerMoveHandle !== null) {
    window.cancelAnimationFrame(pointerMoveHandle);
    pointerMoveHandle = null;
  }
  pendingPointerMove = null;
};

const videoFrameCallbacksAvailable =
  typeof video.requestVideoFrameCallback === "function";

const browserInstanceId = (() => {
  if (typeof crypto?.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const origin = Math.round(performance.timeOrigin).toString(36);
  const random = Math.random().toString(36).slice(2);
  return `${origin}-${random}`;
})();

const showStatus = (message, isError = false) => {
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("error", isError);
};

const isFiniteNonnegative = value =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

const browserEventTimestamp = event => {
  const eventTimestamp = event?.timeStamp;
  return isFiniteNonnegative(eventTimestamp) ? eventTimestamp : performance.now();
};

const nextEventId = () => {
  inputSequence += 1;
  return `${browserInstanceId}:${inputSequence.toString(36)}`;
};

const rememberBounded = (mapping, key, value, limit) => {
  if (mapping.has(key)) {
    mapping.delete(key);
  }
  mapping.set(key, value);
  while (mapping.size > limit) {
    mapping.delete(mapping.keys().next().value);
  }
};

const send = (payload, channel = controls) => {
  if (channel.readyState !== "open") {
    return false;
  }
  try {
    channel.send(JSON.stringify(payload));
    return true;
  } catch (error) {
    console.debug("Unable to send WebRTC control message.", error);
    return false;
  }
};

const formatClockTime = browserTimestampMs => {
  if (!isFiniteNonnegative(browserTimestampMs)) {
    return "—";
  }
  const absoluteTime = performance.timeOrigin + browserTimestampMs;
  const date = new Date(absoluteTime);
  if (!Number.isFinite(date.getTime())) {
    return `${browserTimestampMs.toFixed(3)} ms`;
  }
  const pad = (value, width = 2) => String(value).padStart(width, "0");
  return [
    pad(date.getHours()),
    pad(date.getMinutes()),
    `${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`,
  ].join(":");
};

const percentile = quantile => {
  if (!latencySamples.length) {
    return null;
  }
  const values = [...latencySamples].sort((left, right) => left - right);
  const position = (values.length - 1) * quantile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const fraction = position - lowerIndex;
  return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * fraction;
};

const formatLatency = value =>
  value === null ? "—" : `${Math.round(value)} ms`;

const renderLatencyPanel = () => {
  panelRenderHandle = null;
  const record = latestPanelRecord;
  const p50 = percentile(0.5);
  const p90 = percentile(0.9);
  latencyP50.textContent = formatLatency(p50);
  latencyP90.textContent = formatLatency(p90);

  if (record === null) {
    latencyPanel.dataset.state = "unavailable";
    latencyValue.textContent = "— ms";
    latencyState.textContent = videoFrameCallbacksAvailable
      ? "Unavailable"
      : "Unsupported";
    latencyEventLabel.textContent = "NO EVENT";
    latencyEventTime.textContent = "—";
    latencyFrameTime.textContent = "—";
    latencyDetail.textContent = videoFrameCallbacksAvailable
      ? "Waiting for input"
      : "Frame callbacks unavailable";
    return;
  }

  latencyEventLabel.textContent = record.label;
  latencyEventTime.textContent = formatClockTime(record.browserEventAtMs);
  latencyFrameTime.textContent = formatClockTime(record.browserPresentedAtMs);

  if (record.status === "presented") {
    latencyPanel.dataset.state = "ready";
    latencyValue.textContent = `${Math.round(record.latencyMs)} ms`;
    latencyState.textContent = "Presented";
  } else if (record.status === "pending" || record.status === "frame-pending") {
    latencyPanel.dataset.state = "pending";
    latencyValue.textContent = "— ms";
    latencyState.textContent = "Pending";
  } else {
    latencyPanel.dataset.state = "unavailable";
    latencyValue.textContent = "— ms";
    latencyState.textContent = "Unavailable";
  }

  const pending = pendingInputs.size;
  const pendingLabel = `${pending} pending`;
  latencyDetail.textContent = record.detail
    ? `${record.detail} · ${pendingLabel}`
    : pendingLabel;
};

const scheduleLatencyPanelRender = () => {
  if (panelRenderHandle === null) {
    panelRenderHandle = window.requestAnimationFrame(renderLatencyPanel);
  }
};

const completeEvent = eventId => {
  pendingInputs.delete(eventId);
  rememberBounded(
    completedEventIds,
    eventId,
    performance.now(),
    MAX_COMPLETED_IDS,
  );
};

const markEventUnavailable = (eventId, detail) => {
  if (completedEventIds.has(eventId)) {
    return;
  }
  const record = pendingInputs.get(eventId);
  if (record === undefined) {
    return;
  }
  record.status = "unavailable";
  record.detail = detail;
  record.browserPresentedAtMs = null;
  record.latencyMs = null;
  completeEvent(eventId);
  if (latestPanelRecord === record) {
    scheduleLatencyPanelRender();
  }
};

const expireMarker = (marker, detail) => {
  for (const trace of marker.traces) {
    markEventUnavailable(trace.eventId, detail);
  }
};

const pruneCaches = (force = false) => {
  const now = performance.now();
  const overCapacity = pendingInputs.size > MAX_PENDING_INPUTS ||
    frameMarkers.size > MAX_FRAME_CACHE || renderedFrames.size > MAX_FRAME_CACHE;
  if (!force && !overCapacity && now - lastCachePruneAtMs < 1000) {
    return;
  }
  lastCachePruneAtMs = now;
  for (const [eventId, record] of pendingInputs) {
    if (now - record.createdAtMs > INPUT_TRACE_TTL_MS) {
      markEventUnavailable(eventId, "No processed frame was reported");
    }
  }
  while (pendingInputs.size > MAX_PENDING_INPUTS) {
    const eventId = pendingInputs.keys().next().value;
    markEventUnavailable(eventId, "Input trace cache was full");
  }

  for (const [frameId, marker] of frameMarkers) {
    if (now - marker.arrivedAtMs > FRAME_TRACE_TTL_MS) {
      frameMarkers.delete(frameId);
      expireMarker(
        marker,
        marker.ambiguous
          ? "Frame correlation was ambiguous"
          : "The processed frame was not displayed",
      );
    }
  }
  while (frameMarkers.size > MAX_FRAME_CACHE) {
    const frameId = frameMarkers.keys().next().value;
    const marker = frameMarkers.get(frameId);
    frameMarkers.delete(frameId);
    expireMarker(marker, "Frame marker cache was full");
  }

  for (const [callbackId, frame] of renderedFrames) {
    if (now - frame.arrivedAtMs > FRAME_TRACE_TTL_MS) {
      renderedFrames.delete(callbackId);
    }
  }
  while (renderedFrames.size > MAX_FRAME_CACHE) {
    renderedFrames.delete(renderedFrames.keys().next().value);
  }
};

const sendInput = (
  payload,
  event,
  label,
  {
    channel = controls,
    trackLatency = false,
    dropIfCongested = false,
  } = {},
) => {
  if (dropIfCongested && channel.bufferedAmount > MAX_NONCRITICAL_BUFFER_BYTES) {
    return null;
  }
  if (!trackLatency) {
    return send(payload, channel);
  }
  const eventId = nextEventId();
  const browserEventAtMs = browserEventTimestamp(event);
  const wasSent = send({
    ...payload,
    event_id: eventId,
  }, channel);
  if (!wasSent) {
    return null;
  }
  const record = {
    eventId,
    browserEventAtMs,
    browserPresentedAtMs: null,
    createdAtMs: performance.now(),
    detail: videoFrameCallbacksAvailable
      ? "Waiting for an acknowledged frame"
      : "Frame callbacks unavailable",
    label,
    latencyMs: null,
    status: videoFrameCallbacksAvailable ? "pending" : "unavailable",
  };
  pendingInputs.set(eventId, record);
  if (!videoFrameCallbacksAvailable) {
    completeEvent(eventId);
  }
  latestPanelRecord = record;
  pruneCaches();
  scheduleLatencyPanelRender();
  return eventId;
};

const uint32 = value => {
  let parsed = value;
  if (typeof parsed === "string" && /^\d+$/.test(parsed)) {
    parsed = Number(parsed);
  }
  if (
    typeof parsed !== "number" ||
    !Number.isInteger(parsed) ||
    parsed < 0 ||
    parsed >= UINT32_MODULUS
  ) {
    return null;
  }
  return parsed;
};

const addUint32 = (left, right) => (left + right) % UINT32_MODULUS;
const subtractUint32 = (left, right) =>
  (left - right + UINT32_MODULUS) % UINT32_MODULUS;

const normalizeFrameMarker = payload => {
  const frameId = payload.frame_id;
  if (!Number.isSafeInteger(frameId) || frameId < 0) {
    return null;
  }
  const framePts = payload.frame_pts;
  const timeBaseNumerator = payload.time_base_num;
  const timeBaseDenominator = payload.time_base_den;
  if (
    !Number.isSafeInteger(framePts) ||
    framePts < 0 ||
    !Number.isSafeInteger(timeBaseNumerator) ||
    timeBaseNumerator <= 0 ||
    !Number.isSafeInteger(timeBaseDenominator) ||
    timeBaseDenominator <= 0 ||
    !Array.isArray(payload.traces)
  ) {
    return null;
  }
  const sourceRtpTimestamp =
    payload.source_rtp_timestamp === null ||
    payload.source_rtp_timestamp === undefined
      ? null
      : uint32(payload.source_rtp_timestamp);
  if (
    payload.source_rtp_timestamp !== null &&
    payload.source_rtp_timestamp !== undefined &&
    sourceRtpTimestamp === null
  ) {
    return null;
  }

  const seenEventIds = new Set();
  const traces = [];
  for (const trace of payload.traces) {
    const eventId = trace?.event_id;
    if (typeof eventId !== "string" || !eventId || seenEventIds.has(eventId)) {
      continue;
    }
    seenEventIds.add(eventId);
    traces.push({eventId});
  }
  const mediaTime = framePts * timeBaseNumerator / timeBaseDenominator;
  if (!Number.isFinite(mediaTime) || mediaTime < 0) {
    return null;
  }
  const fingerprint = JSON.stringify({
    framePts,
    sourceRtpTimestamp,
    timeBaseDenominator,
    timeBaseNumerator,
    traces: traces.map(trace => trace.eventId),
  });
  return {
    ambiguous: false,
    arrivedAtMs: performance.now(),
    fingerprint,
    frameId,
    framePts,
    mediaTime,
    sourceRtpTimestamp,
    timeBaseDenominator,
    timeBaseNumerator,
    traces,
  };
};

const markMarkerPending = marker => {
  for (const trace of marker.traces) {
    const record = pendingInputs.get(trace.eventId);
    if (record === undefined) {
      continue;
    }
    record.status = "frame-pending";
    record.detail = `Frame ${marker.frameId} tagged; waiting for display`;
    if (latestPanelRecord === record) {
      scheduleLatencyPanelRender();
    }
  }
};

const markMarkerAmbiguous = marker => {
  marker.ambiguous = true;
  for (const trace of marker.traces) {
    const record = pendingInputs.get(trace.eventId);
    if (record === undefined) {
      continue;
    }
    record.status = "ambiguous";
    record.detail = "Frame correlation is ambiguous";
    if (latestPanelRecord === record) {
      scheduleLatencyPanelRender();
    }
  }
};

const normalizeRenderedFrame = (now, metadata) => {
  const mediaTime = isFiniteNonnegative(metadata.mediaTime)
    ? metadata.mediaTime
    : null;
  const rtpTimestamp = uint32(metadata.rtpTimestamp);
  const presentedFrames = isFiniteNonnegative(metadata.presentedFrames)
    ? metadata.presentedFrames
    : null;
  const expectedDisplayTimeIsPresent =
    metadata.expectedDisplayTime !== undefined;
  const browserPresentedAtMs = expectedDisplayTimeIsPresent
    ? (
      isFiniteNonnegative(metadata.expectedDisplayTime)
        ? metadata.expectedDisplayTime
        : null
    )
    : (
      isFiniteNonnegative(metadata.presentationTime)
        ? metadata.presentationTime
        : null
    );
  const signature = JSON.stringify({
    mediaTime,
    presentedFrames,
    presentationTime: metadata.presentationTime,
    rtpTimestamp,
  });
  if (renderedFrameSignatures.has(signature)) {
    return null;
  }
  rememberBounded(
    renderedFrameSignatures,
    signature,
    performance.now(),
    MAX_COMPLETED_IDS,
  );
  renderedFrameSequence += 1;
  return {
    arrivedAtMs: performance.now(),
    browserPresentedAtMs,
    callbackAtMs: now,
    callbackId: String(renderedFrameSequence),
    mediaTime,
    presentedFrames,
    rtpTimestamp,
  };
};

const mediaTimeMatches = (marker, renderedFrame) => {
  if (renderedFrame.mediaTime === null) {
    return false;
  }
  if (
    rtpTimestampOffset !== null &&
    marker.sourceRtpTimestamp !== null &&
    renderedFrame.rtpTimestamp !== null
  ) {
    return false;
  }
  const tickSeconds = marker.timeBaseNumerator / marker.timeBaseDenominator;
  const toleranceSeconds = Math.max(0.00005, Math.min(0.002, tickSeconds * 0.1));
  return Math.abs(marker.mediaTime - renderedFrame.mediaTime) <= toleranceSeconds;
};

const rtpTimestampMatches = (marker, renderedFrame) => {
  if (
    rtpTimestampOffset === null ||
    marker.sourceRtpTimestamp === null ||
    renderedFrame.rtpTimestamp === null
  ) {
    return false;
  }
  const expectedTimestamp = addUint32(
    marker.sourceRtpTimestamp,
    rtpTimestampOffset,
  );
  const forwardDistance = subtractUint32(
    renderedFrame.rtpTimestamp,
    expectedTimestamp,
  );
  const backwardDistance = subtractUint32(
    expectedTimestamp,
    renderedFrame.rtpTimestamp,
  );
  return Math.min(forwardDistance, backwardDistance) <=
    RTP_TIMESTAMP_TOLERANCE_TICKS;
};

const mutualUniquePair = predicate => {
  const markers = [...frameMarkers.values()];
  const frames = [...renderedFrames.values()];
  for (const marker of markers) {
    const matchingFrames = frames.filter(frame => predicate(marker, frame));
    if (matchingFrames.length > 1) {
      markMarkerAmbiguous(marker);
      continue;
    }
    if (matchingFrames.length !== 1) {
      continue;
    }
    const renderedFrame = matchingFrames[0];
    const matchingMarkers = markers.filter(candidate =>
      predicate(candidate, renderedFrame),
    );
    if (matchingMarkers.length > 1) {
      for (const candidate of matchingMarkers) {
        markMarkerAmbiguous(candidate);
      }
      continue;
    }
    return [marker, renderedFrame];
  }
  return null;
};

const calibrateRtpTimestamp = (marker, renderedFrame) => {
  if (
    rtpCalibrationConflicted ||
    marker.sourceRtpTimestamp === null ||
    renderedFrame.rtpTimestamp === null
  ) {
    return;
  }
  const observedOffset = subtractUint32(
    renderedFrame.rtpTimestamp,
    marker.sourceRtpTimestamp,
  );
  if (rtpTimestampOffset === null) {
    rtpTimestampOffset = observedOffset;
    return;
  }
  if (rtpTimestampOffset !== observedOffset) {
    rtpTimestampOffset = null;
    rtpCalibrationConflicted = true;
    for (const pendingMarker of frameMarkers.values()) {
      if (pendingMarker.sourceRtpTimestamp !== null) {
        markMarkerAmbiguous(pendingMarker);
      }
    }
  }
};

const recordLatency = (marker, renderedFrame, trace) => {
  const eventId = trace.eventId;
  if (completedEventIds.has(eventId)) {
    return;
  }
  const record = pendingInputs.get(eventId);
  if (record === undefined) {
    rememberBounded(
      completedEventIds,
      eventId,
      performance.now(),
      MAX_COMPLETED_IDS,
    );
    return;
  }
  const browserPresentedAtMs = renderedFrame.browserPresentedAtMs;
  if (!isFiniteNonnegative(browserPresentedAtMs)) {
    markEventUnavailable(eventId, "Browser presentation timestamp unavailable");
    return;
  }
  const latencyMs = browserPresentedAtMs - record.browserEventAtMs;
  if (!Number.isFinite(latencyMs) || latencyMs < 0) {
    markEventUnavailable(eventId, "Browser timestamps could not be compared");
    return;
  }

  record.status = "presented";
  record.detail = `Event ${eventId.split(":").at(-1)} · frame ${marker.frameId}`;
  record.browserPresentedAtMs = browserPresentedAtMs;
  record.latencyMs = latencyMs;
  completeEvent(eventId);
  latencySamples.push(latencyMs);
  if (latencySamples.length > MAX_LATENCY_SAMPLES) {
    latencySamples.splice(0, latencySamples.length - MAX_LATENCY_SAMPLES);
  }
  latestPanelRecord = record;
};

const resolveFramePair = (marker, renderedFrame, matchedByMediaTime) => {
  frameMarkers.delete(marker.frameId);
  renderedFrames.delete(renderedFrame.callbackId);
  rememberBounded(
    completedFrameIds,
    marker.frameId,
    performance.now(),
    MAX_COMPLETED_IDS,
  );
  if (matchedByMediaTime) {
    calibrateRtpTimestamp(marker, renderedFrame);
  }
  for (const trace of marker.traces) {
    recordLatency(marker, renderedFrame, trace);
  }
  scheduleLatencyPanelRender();
};

const reconcileFrames = () => {
  pruneCaches();
  while (true) {
    if (rtpTimestampOffset !== null) {
      const rtpPair = mutualUniquePair(rtpTimestampMatches);
      if (rtpPair !== null) {
        resolveFramePair(rtpPair[0], rtpPair[1], false);
        continue;
      }
    }
    const mediaPair = mutualUniquePair(mediaTimeMatches);
    if (mediaPair !== null) {
      resolveFramePair(mediaPair[0], mediaPair[1], true);
      continue;
    }
    break;
  }
};

const handleInputFrame = payload => {
  const marker = normalizeFrameMarker(payload);
  if (marker === null) {
    console.warn("Ignored malformed input-frame marker.", payload);
    return;
  }
  if (completedFrameIds.has(marker.frameId)) {
    return;
  }
  const existing = frameMarkers.get(marker.frameId);
  if (existing !== undefined) {
    if (existing.fingerprint === marker.fingerprint) {
      return;
    }
    frameMarkers.delete(marker.frameId);
    expireMarker(existing, "Conflicting frame markers were received");
    expireMarker(marker, "Conflicting frame markers were received");
    rememberBounded(
      completedFrameIds,
      marker.frameId,
      performance.now(),
      MAX_COMPLETED_IDS,
    );
    return;
  }
  if (!videoFrameCallbacksAvailable) {
    expireMarker(marker, "Frame callbacks unavailable");
    rememberBounded(
      completedFrameIds,
      marker.frameId,
      performance.now(),
      MAX_COMPLETED_IDS,
    );
    return;
  }
  frameMarkers.set(marker.frameId, marker);
  markMarkerPending(marker);
  reconcileFrames();
};

const handleDroppedInputFrame = payload => {
  if (!Array.isArray(payload.event_ids)) {
    return;
  }
  for (const eventId of payload.event_ids) {
    if (typeof eventId === "string" && eventId) {
      markEventUnavailable(eventId, "The processed frame was dropped before encoding");
    }
  }
};

const handleInputTraceReset = () => {
  for (const eventId of [...pendingInputs.keys()]) {
    markEventUnavailable(eventId, "Input trace updates overflowed on the server");
  }
};

const failPendingInputs = detail => {
  for (const eventId of [...pendingInputs.keys()]) {
    markEventUnavailable(eventId, detail);
  }
  frameMarkers.clear();
  renderedFrames.clear();
};

controls.addEventListener("message", event => {
  if (typeof event.data !== "string") {
    return;
  }
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (error) {
    console.warn("Ignored malformed WebRTC control response.", error);
    return;
  }
  if (payload?.type === "input_frame") {
    handleInputFrame(payload);
  } else if (payload?.type === "input_frame_dropped") {
    handleDroppedInputFrame(payload);
  } else if (payload?.type === "input_trace_reset") {
    handleInputTraceReset();
  } else if (payload?.type === "error") {
    console.warn(`WebRTC server: ${payload.message ?? "unknown error"}`);
  }
});

controls.addEventListener("close", () => {
  failPendingInputs("The WebRTC control channel closed before presentation");
});

const observeVideoFrames = () => {
  if (!videoFrameCallbacksAvailable) {
    renderLatencyPanel();
    return;
  }
  const onFrame = (now, metadata) => {
    try {
      const renderedFrame = normalizeRenderedFrame(now, metadata);
      if (renderedFrame !== null) {
        renderedFrames.set(renderedFrame.callbackId, renderedFrame);
        reconcileFrames();
      }
    } finally {
      video.requestVideoFrameCallback(onFrame);
    }
  };
  video.requestVideoFrameCallback(onFrame);
};

peer.ontrack = event => {
  video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
  video.play().catch(error => {
    showStatus(`Video playback failed: ${error.message}`, true);
  });
};

video.addEventListener("playing", () => {
  status.hidden = true;
});

peer.addEventListener("connectionstatechange", () => {
  if (peer.connectionState === "connected") {
    if (video.readyState < 2) {
      showStatus("Connected. Waiting for the first video frame…");
    } else {
      status.hidden = true;
    }
  } else if (["failed", "closed"].includes(peer.connectionState)) {
    showStatus(`WebRTC connection ${peer.connectionState}.`, true);
    failPendingInputs(`WebRTC connection ${peer.connectionState}`);
  }
});

window.addEventListener("keydown", event => {
  const keyId = event.code || event.key;
  if (pressedKeys.has(keyId)) {
    return;
  }
  const eventId = sendInput(
    {type: "keyboard", key: event.key, pressed: true},
    event,
    `${event.key.toUpperCase()} DOWN`,
    {trackLatency: true},
  );
  if (eventId !== null) {
    pressedKeys.set(keyId, event.key);
  }
});

window.addEventListener("keyup", event => {
  const keyId = event.code || event.key;
  const pressedKey = pressedKeys.get(keyId);
  if (pressedKey === undefined) {
    return;
  }
  const eventId = sendInput(
    {type: "keyboard", key: pressedKey, pressed: false},
    event,
    `${pressedKey.toUpperCase()} UP`,
    {trackLatency: true},
  );
  if (eventId !== null) {
    pressedKeys.delete(keyId);
  }
});

video.tabIndex = 0;

const renderedVideoBounds = () => {
  const bounds = video.getBoundingClientRect();
  if (!video.videoWidth || !video.videoHeight || !bounds.width || !bounds.height) {
    return bounds;
  }

  const scale = Math.min(
    bounds.width / video.videoWidth,
    bounds.height / video.videoHeight,
  );
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  return {
    left: bounds.left + (bounds.width - width) / 2,
    top: bounds.top + (bounds.height - height) / 2,
    width,
    height,
  };
};

const pointerPosition = event => {
  const bounds = renderedVideoBounds();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
    y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
  };
};

video.addEventListener("pointermove", event => {
  lastPointerPosition = pointerPosition(event);
  pendingPointerMove = {
    position: lastPointerPosition,
    timestamp: browserEventTimestamp(event),
  };
  if (pointerMoveHandle !== null) {
    return;
  }
  pointerMoveHandle = window.requestAnimationFrame(() => {
    pointerMoveHandle = null;
    const move = pendingPointerMove;
    pendingPointerMove = null;
    if (move === null) {
      return;
    }
    sendInput(
      {
        type: "mouse",
        action: "move",
        ...move.position,
      },
      {timeStamp: move.timestamp},
      "POINTER MOVE",
      {
        channel: pointerControls,
        dropIfCongested: true,
      },
    );
  });
});

video.addEventListener("pointerdown", event => {
  video.focus();
  video.setPointerCapture(event.pointerId);
  cancelPendingPointerMove();
  lastPointerPosition = pointerPosition(event);
  const wasSent = sendInput({
    type: "mouse",
    action: "button",
    ...lastPointerPosition,
    button: event.button,
    pressed: true,
  }, event, `BUTTON ${event.button} DOWN`, {channel: pointerControls});
  if (wasSent) {
    pressedButtons.add(event.button);
  }
  event.preventDefault();
});

video.addEventListener("pointerup", event => {
  if (!pressedButtons.has(event.button)) {
    return;
  }
  cancelPendingPointerMove();
  lastPointerPosition = pointerPosition(event);
  const wasSent = sendInput({
    type: "mouse",
    action: "button",
    ...lastPointerPosition,
    button: event.button,
    pressed: false,
  }, event, `BUTTON ${event.button} UP`, {channel: pointerControls});
  if (wasSent) {
    pressedButtons.delete(event.button);
  }
  event.preventDefault();
});

video.addEventListener("pointercancel", event => {
  cancelPendingPointerMove();
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput({
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button,
      pressed: false,
    }, event, `BUTTON ${button} CANCEL`, {channel: pointerControls});
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
});

video.addEventListener("wheel", event => {
  const position = pointerPosition(event);
  if (pendingWheel === null) {
    pendingWheel = {
      position,
      timestamp: browserEventTimestamp(event),
      wheelX: 0,
      wheelY: 0,
    };
  }
  pendingWheel.position = position;
  pendingWheel.timestamp = browserEventTimestamp(event);
  pendingWheel.wheelX += -Math.sign(event.deltaX);
  pendingWheel.wheelY += -Math.sign(event.deltaY);
  if (wheelHandle === null) {
    wheelHandle = window.requestAnimationFrame(() => {
      wheelHandle = null;
      const wheel = pendingWheel;
      pendingWheel = null;
      if (wheel === null) {
        return;
      }
      sendInput({
        type: "mouse",
        action: "wheel",
        ...wheel.position,
        wheel_x: wheel.wheelX,
        wheel_y: wheel.wheelY,
      }, {timeStamp: wheel.timestamp}, "WHEEL", {
        channel: pointerControls,
        dropIfCongested: true,
      });
    });
  }
  event.preventDefault();
}, {passive: false});

video.addEventListener("focus", event => {
  sendInput({type: "focus", focused: true}, event, "FOCUS IN");
});
video.addEventListener("blur", event => {
  sendInput({type: "focus", focused: false}, event, "FOCUS OUT");
});

window.addEventListener("blur", event => {
  cancelPendingPointerMove();
  for (const [keyId, key] of [...pressedKeys]) {
    const wasSent = sendInput(
      {type: "keyboard", key, pressed: false},
      event,
      `${key.toUpperCase()} RELEASE`,
    );
    if (wasSent) {
      pressedKeys.delete(keyId);
    }
  }
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput({
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button,
      pressed: false,
    }, event, `BUTTON ${button} RELEASE`, {channel: pointerControls});
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
});

window.addEventListener("beforeunload", event => {
  sendInput({type: "close"}, event, "CLOSE");
});

const waitForIceGatheringComplete = async () => {
  if (peer.iceGatheringState === "complete") {
    return;
  }
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", onStateChange);
      reject(new Error("Timed out while gathering WebRTC network candidates."));
    }, 10000);
    const onStateChange = () => {
      if (peer.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        peer.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", onStateChange);
  });
};

const waitForServer = async () => {
  while (true) {
    try {
      const health = await fetch("/healthz", {cache: "no-store"});
      if (health.ok && (await health.json()).open) {
        return;
      }
    } catch (error) {
      console.debug("WebRTC server is not ready yet.", error);
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
};

async function connect() {
  showStatus("Waiting for the server…");
  await waitForServer();
  showStatus("Gathering WebRTC network candidates…");
  await peer.setLocalDescription(await peer.createOffer());
  await waitForIceGatheringComplete();
  const response = await fetch("/api/webrtc/offer", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(peer.localDescription),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  showStatus("Connecting video…");
  await peer.setRemoteDescription(await response.json());
}

observeVideoFrames();
connect().catch(error => {
  console.error("Unable to start WebRTC.", error);
  showStatus(`Unable to start WebRTC: ${error.message}`, true);
});
