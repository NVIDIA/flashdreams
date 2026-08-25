// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const peer = new RTCPeerConnection();
const controls = peer.createDataChannel("controls");
peer.addTransceiver("video", {direction: "recvonly"});
const video = document.getElementById("video");
const status = document.getElementById("status");
const pressedKeys = new Set();
const pressedButtons = new Set();
const gamepadSnapshots = new Map();
let lastPointerPosition = {x: 0, y: 0};

const showStatus = (message, isError = false) => {
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("error", isError);
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
  if (peer.connectionState === "connected" && video.readyState < 2) {
    showStatus("Connected. Waiting for the first video frame…");
  } else if (["failed", "disconnected", "closed"].includes(peer.connectionState)) {
    showStatus(`WebRTC connection ${peer.connectionState}.`, true);
  }
});

const send = payload => {
  if (controls.readyState === "open") {
    controls.send(JSON.stringify(payload));
  }
};

window.addEventListener("keydown", event => {
  pressedKeys.add(event.key);
  send({type: "keyboard", key: event.key, pressed: true});
});

window.addEventListener("keyup", event => {
  pressedKeys.delete(event.key);
  send({type: "keyboard", key: event.key, pressed: false});
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
  send({type: "mouse", action: "move", ...lastPointerPosition});
});

video.addEventListener("pointerdown", event => {
  video.focus();
  video.setPointerCapture(event.pointerId);
  pressedButtons.add(event.button);
  lastPointerPosition = pointerPosition(event);
  send({
    type: "mouse",
    action: "button",
    ...lastPointerPosition,
    button: event.button,
    pressed: true,
  });
  event.preventDefault();
});

video.addEventListener("pointerup", event => {
  pressedButtons.delete(event.button);
  lastPointerPosition = pointerPosition(event);
  send({
    type: "mouse",
    action: "button",
    ...lastPointerPosition,
    button: event.button,
    pressed: false,
  });
  event.preventDefault();
});

video.addEventListener("pointercancel", () => {
  for (const button of pressedButtons) {
    send({
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button,
      pressed: false,
    });
  }
  pressedButtons.clear();
});

video.addEventListener("wheel", event => {
  send({
    type: "mouse",
    action: "wheel",
    ...pointerPosition(event),
    wheel_x: -Math.sign(event.deltaX),
    wheel_y: -Math.sign(event.deltaY),
  });
  event.preventDefault();
}, {passive: false});

video.addEventListener("focus", () => send({type: "focus", focused: true}));
video.addEventListener("blur", () => send({type: "focus", focused: false}));

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
  for (const key of pressedKeys) {
    send({type: "keyboard", key, pressed: false});
  }
  pressedKeys.clear();
  for (const button of pressedButtons) {
    send({
      type: "mouse",
      action: "button",
      ...lastPointerPosition,
      button,
      pressed: false,
    });
  }
  pressedButtons.clear();
});

window.addEventListener("beforeunload", () => send({type: "close"}));

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

connect().catch(error => {
  console.error("Unable to start WebRTC.", error);
  showStatus(`Unable to start WebRTC: ${error.message}`, true);
});
