// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const peer = new RTCPeerConnection();
const controls = peer.createDataChannel("controls");
const pointerControls = peer.createDataChannel("pointer-controls");
peer.addTransceiver("video", {direction: "recvonly"});
const video = document.getElementById("video");
const status = document.getElementById("status");
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

connect().catch(error => {
  console.error("Unable to start WebRTC.", error);
  showStatus(`Unable to start WebRTC: ${error.message}`, true);
});
