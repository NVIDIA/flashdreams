// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const peer = new RTCPeerConnection();
const controls = peer.createDataChannel("controls");
const pointerControls = peer.createDataChannel("pointer-controls");
peer.addTransceiver("video", {direction: "recvonly"});
const video = document.getElementById("video");
const status = document.getElementById("status");
const promptDialog = document.getElementById("prompt-dialog");
const promptBackdrop = document.getElementById("prompt-backdrop");
const promptDialogTitle = document.getElementById("prompt-dialog-title");
const promptDialogDetails = document.getElementById("prompt-dialog-details");
const promptDialogConfirm = document.getElementById("prompt-dialog-confirm");
const promptDialogCancel = document.getElementById("prompt-dialog-cancel");
const pressedKeys = new Map();
const pressedButtons = new Set();
const gamepadSnapshots = new Map();
let lastPointerPosition = {x: 0, y: 0};
let hideCursor = false;
let lockCursorToWindow = false;

const MAX_NONCRITICAL_BUFFER_BYTES = 4 * 1024;

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

const showStatus = (message, isError = false) => {
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("error", isError);
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

const sendInput = (
  payload,
  {
    channel = controls,
    dropIfCongested = false,
  } = {},
) => {
  if (dropIfCongested && channel.bufferedAmount > MAX_NONCRITICAL_BUFFER_BYTES) {
    return false;
  }
  return send(payload, channel);
};

const updateCursorVisibility = () => {
  video.style.cursor = hideCursor && document.hasFocus() ? "none" : "";
};

const applyCursorOptions = options => {
  hideCursor = options.hide_cursor === true;
  lockCursorToWindow = options.lock_cursor_to_window === true;
  updateCursorVisibility();
  if (!lockCursorToWindow && document.pointerLockElement === video) {
    document.exitPointerLock();
  }
};

let promptSettled = false;
let promptOnConfirm = null;
let promptOnCancel = null;

const setPromptBackdropVisible = visible => {
  if (promptBackdrop !== null) {
    promptBackdrop.hidden = !visible;
  }
};

const closePromptDialog = () => {
  setPromptBackdropVisible(false);
  if (promptDialog !== null && promptDialog.open) {
    promptDialog.close();
  }
};

const fillPromptDetails = details => {
  if (promptDialogDetails === null) {
    return;
  }
  promptDialogDetails.replaceChildren();
  if (!Array.isArray(details) || details.length === 0) {
    promptDialogDetails.hidden = true;
    return;
  }
  for (const section of details) {
    const row = document.createElement("div");
    row.className = "prompt-dialog-row";
    const label = document.createElement("div");
    label.className = "prompt-dialog-detail-label";
    label.textContent = section.label;
    const value = document.createElement("div");
    value.className = "prompt-dialog-row-value";
    const values = Array.isArray(section.items) ? section.items : [];
    for (const item of values) {
      const chip = document.createElement("span");
      chip.className = "prompt-dialog-chip";
      chip.textContent = item;
      value.append(chip);
    }
    row.append(label, value);
    promptDialogDetails.append(row);
  }
  promptDialogDetails.hidden = false;
};

const showPromptDialog = ({
  title,
  details = [],
  confirmLabel = "OK",
  cancelLabel = "Cancel",
  onConfirm = null,
  onCancel = null,
} = {}) => {
  if (
    promptDialog === null
    || promptDialogTitle === null
    || promptDialogConfirm === null
    || promptDialogCancel === null
  ) {
    return;
  }
  if (promptDialog.open) {
    promptSettled = true;
    promptOnConfirm = null;
    promptOnCancel = null;
    promptDialog.close();
  }
  promptDialogTitle.textContent = title;
  fillPromptDetails(details);
  promptDialogConfirm.textContent = confirmLabel;
  promptDialogCancel.textContent = cancelLabel;
  promptOnConfirm = onConfirm;
  promptOnCancel = onCancel;
  promptSettled = false;
  setPromptBackdropVisible(true);
  promptDialog.show();
  promptDialogConfirm.focus();
};

if (promptDialog !== null) {
  promptDialogConfirm?.addEventListener("click", () => {
    if (promptSettled) {
      return;
    }
    promptSettled = true;
    const onConfirm = promptOnConfirm;
    promptOnConfirm = null;
    promptOnCancel = null;
    closePromptDialog();
    onConfirm?.();
  });
  promptDialogCancel?.addEventListener("click", () => {
    if (promptSettled) {
      return;
    }
    promptSettled = true;
    const onCancel = promptOnCancel;
    promptOnConfirm = null;
    promptOnCancel = null;
    closePromptDialog();
    onCancel?.();
  });
  promptDialog.addEventListener("close", () => {
    setPromptBackdropVisible(false);
    if (promptSettled) {
      return;
    }
    promptSettled = true;
    const onCancel = promptOnCancel;
    promptOnConfirm = null;
    promptOnCancel = null;
    onCancel?.();
  });
}

window.addEventListener("keydown", event => {
  if (event.key !== "Escape" || promptDialog === null || !promptDialog.open) {
    return;
  }
  event.preventDefault();
  promptDialogCancel?.click();
});

if (promptBackdrop !== null) {
  promptBackdrop.addEventListener("click", () => {
    promptDialogCancel?.click();
  });
}

let pendingFileRequest = null;
let queuedFileSelectors = [];

const hideFilePicker = () => {
  pendingFileRequest = null;
  if (promptDialog !== null && promptDialog.open) {
    promptSettled = true;
    promptOnConfirm = null;
    promptOnCancel = null;
    closePromptDialog();
  }
  const next = queuedFileSelectors.shift();
  if (next !== undefined) {
    // The runtime may arm the next selector before this viewer's fetch
    // callback runs; show it after the current dialog has fully settled.
    window.queueMicrotask(() => openFileSelector(next));
  }
};

const completeFileSelection = (requestId, status) => {
  send({type: "file_selector_result", id: requestId, status});
};

const formatByteBudget = bytes => {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "the allowed size";
  }
  const mib = bytes / (1024 * 1024);
  if (mib >= 1) {
    return Number.isInteger(mib) ? `${mib} MiB` : `${mib.toFixed(1)} MiB`;
  }
  const kib = bytes / 1024;
  if (kib >= 1) {
    return Number.isInteger(kib) ? `${kib} KiB` : `${kib.toFixed(1)} KiB`;
  }
  return `${bytes} bytes`;
};

const fileNameMatchesAccept = (name, accept) => {
  if (!Array.isArray(accept) || accept.length === 0) {
    return true;
  }
  const lower = name.toLowerCase();
  return accept.some(
    suffix => typeof suffix === "string" && lower.endsWith(suffix.toLowerCase()),
  );
};

const selectedFilePolicyStatus = (name, size, accept, maxBytes) => {
  // Same order as selected_file_policy_status: type, then size.
  if (!fileNameMatchesAccept(name, accept)) {
    return "disallowed_type";
  }
  if (size > maxBytes) {
    return "too_large";
  }
  return null;
};

const startFileInput = pending => {
  const requestId = pending.id;
  const input = document.createElement("input");
  input.type = "file";
  if (pending.accept.length > 0) {
    input.accept = pending.accept.join(",");
  }
  let settled = false;
  const finishFileSelection = status => {
    if (settled || pendingFileRequest === null || pendingFileRequest.id !== requestId) {
      return;
    }
    settled = true;
    completeFileSelection(requestId, status);
    hideFilePicker();
  };
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      finishFileSelection("cancelled");
      return;
    }
    const rejected = selectedFilePolicyStatus(
      file.name || "",
      file.size,
      pending.accept,
      pending.maxBytes,
    );
    if (rejected !== null) {
      finishFileSelection(rejected);
      return;
    }
    const body = new FormData();
    body.append("file", file, file.name);
    fetch(`/api/files?request_id=${encodeURIComponent(requestId)}`, {
      method: "POST",
      body,
    })
      .then(response => {
        if (settled || pendingFileRequest === null || pendingFileRequest.id !== requestId) {
          return;
        }
        settled = true;
        hideFilePicker();
        if (!response.ok) {
          console.debug("File upload was rejected.", response.status);
        }
      })
      .catch(error => {
        console.debug("Unable to upload the selected file.", error);
        finishFileSelection("unavailable");
      });
  });
  input.addEventListener("cancel", () => finishFileSelection("cancelled"));
  input.click();
};

const openFileSelector = options => {
  const requestId = options?.id;
  if (typeof requestId !== "string" || !requestId) {
    return;
  }
  if (pendingFileRequest !== null) {
    if (
      pendingFileRequest.id !== requestId
      && queuedFileSelectors.every(item => item.id !== requestId)
    ) {
      queuedFileSelectors.push(options);
    }
    return;
  }
  const accept = Array.isArray(options.accept) ? options.accept : [];
  const maxBytes = Number(options.max_bytes);
  pendingFileRequest = {
    id: requestId,
    accept,
    maxBytes: Number.isFinite(maxBytes) ? maxBytes : 0,
  };
  showPromptDialog({
    title: "The app requested a file",
    details: [
      {
        label: "Accepted types",
        items: accept.length > 0 ? accept : ["Any"],
      },
      {
        label: "Max size allowed",
        items: [formatByteBudget(pendingFileRequest.maxBytes)],
      },
    ],
    confirmLabel: "Select…",
    onConfirm: () => {
      const pending = pendingFileRequest;
      if (pending === null || pending.id !== requestId) {
        return;
      }
      startFileInput(pending);
    },
    onCancel: () => {
      if (pendingFileRequest === null || pendingFileRequest.id !== requestId) {
        return;
      }
      completeFileSelection(requestId, "cancelled");
      hideFilePicker();
    },
  });
};

window.addEventListener("focus", updateCursorVisibility);
window.addEventListener("blur", updateCursorVisibility);

controls.addEventListener("message", event => {
  if (typeof event.data !== "string") {
    return;
  }
  try {
    const payload = JSON.parse(event.data);
    if (payload?.type === "cursor_options") {
      applyCursorOptions(payload);
      return;
    }
    if (payload?.type === "file_selector") {
      openFileSelector(payload);
      return;
    }
    if (payload?.type === "error") {
      console.warn(`WebRTC server: ${payload.message ?? "unknown error"}`);
    }
  } catch (error) {
    console.warn("Ignored malformed WebRTC control response.", error);
  }
});

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
  }
});

window.addEventListener("keydown", event => {
  const keyId = event.code || event.key;
  if (pressedKeys.has(keyId)) {
    return;
  }
  const wasSent = sendInput({
    type: "keyboard",
    key: event.key,
    pressed: true,
  });
  if (wasSent) {
    pressedKeys.set(keyId, event.key);
  }
});

window.addEventListener("keyup", event => {
  const keyId = event.code || event.key;
  const pressedKey = pressedKeys.get(keyId);
  if (pressedKey === undefined) {
    return;
  }
  const wasSent = sendInput({
    type: "keyboard",
    key: pressedKey,
    pressed: false,
  });
  if (wasSent) {
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

const activePointerPosition = event => {
  if (!lockCursorToWindow || document.pointerLockElement !== video) {
    return pointerPosition(event);
  }
  const bounds = renderedVideoBounds();
  return {
    x: lastPointerPosition.x + event.movementX / bounds.width,
    y: lastPointerPosition.y + event.movementY / bounds.height,
  };
};

const lockPointerToWindow = () => {
  if (!lockCursorToWindow || document.pointerLockElement === video) {
    return;
  }
  try {
    const request = video.requestPointerLock();
    request?.catch(error => {
      console.debug("Unable to lock the cursor to the window.", error);
    });
  } catch (error) {
    console.debug("Unable to lock the cursor to the window.", error);
  }
};

video.addEventListener("pointermove", event => {
  lastPointerPosition = activePointerPosition(event);
  pendingPointerMove = lastPointerPosition;
  if (pointerMoveHandle !== null) {
    return;
  }
  pointerMoveHandle = window.requestAnimationFrame(() => {
    pointerMoveHandle = null;
    const position = pendingPointerMove;
    pendingPointerMove = null;
    if (position === null) {
      return;
    }
    sendInput(
      {
        type: "mouse",
        action: "move",
        ...position,
      },
      {
        channel: pointerControls,
        dropIfCongested: true,
      },
    );
  });
});

video.addEventListener("pointerdown", event => {
  video.focus();
  if (lockCursorToWindow) {
    lockPointerToWindow();
  } else {
    video.setPointerCapture(event.pointerId);
  }
  cancelPendingPointerMove();
  lastPointerPosition = activePointerPosition(event);
  const wasSent = sendInput(
    {
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button: event.button,
      pressed: true,
    },
    {channel: pointerControls},
  );
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
  lastPointerPosition = activePointerPosition(event);
  const wasSent = sendInput(
    {
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button: event.button,
      pressed: false,
    },
    {channel: pointerControls},
  );
  if (wasSent) {
    pressedButtons.delete(event.button);
  }
  event.preventDefault();
});

video.addEventListener("pointercancel", () => {
  cancelPendingPointerMove();
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput(
      {
        type: "mouse",
        action: "button",
        ...lastPointerPosition,
        button,
        pressed: false,
      },
      {channel: pointerControls},
    );
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
});

video.addEventListener("wheel", event => {
  const position = activePointerPosition(event);
  if (pendingWheel === null) {
    pendingWheel = {
      position,
      wheelX: 0,
      wheelY: 0,
    };
  }
  pendingWheel.position = position;
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
      sendInput(
        {
          type: "mouse",
          action: "wheel",
          ...wheel.position,
          wheel_x: wheel.wheelX,
          wheel_y: wheel.wheelY,
        },
        {
          channel: pointerControls,
          dropIfCongested: true,
        },
      );
    });
  }
  event.preventDefault();
}, {passive: false});

video.addEventListener("focus", () => {
  sendInput({type: "focus", focused: true});
});
video.addEventListener("blur", () => {
  sendInput({type: "focus", focused: false});
});

const touchPayload = (event, touch, action) => {
  const bounds = renderedVideoBounds();
  return {
    type: "touch",
    action,
    touch_id: touch.identifier,
    x: Math.min(1, Math.max(0, (touch.clientX - bounds.left) / bounds.width)),
    y: Math.min(1, Math.max(0, (touch.clientY - bounds.top) / bounds.height)),
    pressure: Math.min(1, Math.max(0, touch.force || 0)),
    primary: touch.identifier === event.touches[0]?.identifier,
  };
};

for (const [domEvent, action] of [
  ["touchstart", "start"],
  ["touchmove", "move"],
  ["touchend", "end"],
  ["touchcancel", "cancel"],
]) {
  video.addEventListener(domEvent, event => {
    for (const touch of event.changedTouches) {
      send(touchPayload(event, touch, action));
    }
    event.preventDefault();
  }, {passive: false});
}

const gamepadPayload = (gamepad, action = "state") => ({
  type: "gamepad",
  action,
  index: gamepad.index,
  id: gamepad.id,
  mapping: gamepad.mapping,
  axes: Array.from(gamepad.axes),
  buttons: gamepad.buttons.map(button => button.value),
  pressed: gamepad.buttons.map(button => button.pressed),
});

window.addEventListener("gamepadconnected", event => {
  gamepadSnapshots.delete(event.gamepad.index);
  send(gamepadPayload(event.gamepad, "connected"));
});

window.addEventListener("gamepaddisconnected", event => {
  gamepadSnapshots.delete(event.gamepad.index);
  send(gamepadPayload(event.gamepad, "disconnected"));
});

const pollGamepads = () => {
  for (const gamepad of navigator.getGamepads?.() || []) {
    if (!gamepad) {
      continue;
    }
    const payload = gamepadPayload(gamepad);
    const snapshot = JSON.stringify(payload);
    if (gamepadSnapshots.get(gamepad.index) !== snapshot) {
      gamepadSnapshots.set(gamepad.index, snapshot);
      send(payload);
    }
  }
  window.requestAnimationFrame(pollGamepads);
};
window.requestAnimationFrame(pollGamepads);

window.addEventListener("blur", () => {
  cancelPendingPointerMove();
  for (const [keyId, key] of [...pressedKeys]) {
    const wasSent = sendInput({
      type: "keyboard",
      key,
      pressed: false,
    });
    if (wasSent) {
      pressedKeys.delete(keyId);
    }
  }
  for (const button of [...pressedButtons]) {
    const wasSent = sendInput(
      {
        type: "mouse",
        action: "button",
        ...lastPointerPosition,
        button,
        pressed: false,
      },
      {channel: pointerControls},
    );
    if (wasSent) {
      pressedButtons.delete(button);
    }
  }
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
      const state = health.ok ? await health.json() : null;
      if (state?.open) {
        return state;
      }
    } catch (error) {
      console.debug("WebRTC server is not ready yet.", error);
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
};

async function connect() {
  showStatus("Waiting for the server…");
  const server = await waitForServer();
  applyCursorOptions(server);
  showStatus("Gathering WebRTC network candidates…");
  await peer.setLocalDescription(await peer.createOffer());
  await waitForIceGatheringComplete();
  const response = await fetch(
    "/api/webrtc/offer" + window.location.search,
    {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(peer.localDescription),
    },
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  showStatus("Connecting video…");
  await peer.setRemoteDescription(await response.json());
}

connect().catch(error => {
  console.error("Unable to start WebRTC.", error);
  showStatus(`Unable to start WebRTC: ${error.message}`, true);
});
