// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const video = document.getElementById("video");
const status = document.getElementById("status");
const activateButton = document.getElementById("activate");
const resetButton = document.getElementById("reset");
const stopButton = document.getElementById("stop");

let peer = null;
let controls = null;
let active = false;
let stopped = false;
let reconnectTimer = null;
let browserConfig = null;
const heldKeys = new Set();
const clientIdStorageKey = "flashdreams-webrtc-client-id";
let clientId = window.sessionStorage.getItem(clientIdStorageKey);
if (clientId === null) {
  const clientIdBytes = new Uint8Array(16);
  crypto.getRandomValues(clientIdBytes);
  clientId = Array.from(
    clientIdBytes,
    value => value.toString(16).padStart(2, "0"),
  ).join("");
  window.sessionStorage.setItem(clientIdStorageKey, clientId);
}

const endpoint = path => new URL(path, window.location.href).toString();

const setStatus = text => {
  status.textContent = text;
};

const send = payload => {
  if (controls?.readyState === "open") {
    controls.send(JSON.stringify(payload));
    return true;
  }
  return false;
};

const releaseControls = () => {
  for (const key of heldKeys) {
    send({type: "keyboard", key, pressed: false});
  }
  heldKeys.clear();
  if (active) {
    active = false;
    if (browserConfig?.pointerLockControls) {
      send({type: "activation", active: false});
    } else {
      send({type: "keyboard", key: "r", pressed: false});
    }
  }
  activateButton.textContent = "Activate";
};

const setActive = async enabled => {
  if (enabled === active || controls?.readyState !== "open") {
    return;
  }
  if (enabled && browserConfig?.pointerLockControls) {
    try {
      await video.requestPointerLock();
    } catch {
      setStatus("Pointer lock was declined");
      return;
    }
  } else if (
    browserConfig?.pointerLockControls &&
    document.pointerLockElement === video
  ) {
    document.exitPointerLock();
  }
  active = enabled;
  activateButton.textContent = active ? "Deactivate" : "Activate";
  if (browserConfig?.pointerLockControls) {
    send({type: "activation", active});
  } else {
    send({type: "keyboard", key: "r", pressed: active});
  }
};

const sendKey = (key, pressed) => {
  if (browserConfig?.pointerLockControls && !active) {
    return;
  }
  if (pressed) {
    if (heldKeys.has(key)) {
      return;
    }
    heldKeys.add(key);
  } else if (!heldKeys.delete(key)) {
    return;
  }
  send({type: "keyboard", key, pressed});
};

window.addEventListener("keydown", event => {
  if (browserConfig?.pointerLockControls && event.key === "Escape") {
    void setActive(false);
    return;
  }
  sendKey(event.key, true);
  if (browserConfig?.pointerLockControls && active) {
    event.preventDefault();
  }
});

window.addEventListener("keyup", event => {
  sendKey(event.key, false);
  if (browserConfig?.pointerLockControls && active) {
    event.preventDefault();
  }
});

const mouseKey = button => {
  if (button === 0) {
    return "Mouse1";
  }
  if (button === 1) {
    return "Mouse3";
  }
  if (button === 2) {
    return "Mouse2";
  }
  return null;
};

video.addEventListener("mousedown", event => {
  if (!browserConfig?.pointerLockControls) {
    return;
  }
  const key = mouseKey(event.button);
  if (key !== null) {
    sendKey(key, true);
    event.preventDefault();
  }
});

window.addEventListener("mouseup", event => {
  if (!browserConfig?.pointerLockControls) {
    return;
  }
  const key = mouseKey(event.button);
  if (key !== null) {
    sendKey(key, false);
  }
});

video.addEventListener("mousemove", event => {
  if (
    browserConfig?.pointerLockControls &&
    active &&
    document.pointerLockElement === video
  ) {
    send({
      type: "mouse",
      movement_x: event.movementX,
      movement_y: event.movementY,
    });
  }
});

video.addEventListener("contextmenu", event => {
  if (browserConfig?.pointerLockControls) {
    event.preventDefault();
  }
});

document.addEventListener("pointerlockchange", () => {
  if (
    browserConfig?.pointerLockControls &&
    active &&
    document.pointerLockElement !== video
  ) {
    releaseControls();
    setStatus("Paused — pointer lock released");
  }
});

window.addEventListener("blur", () => {
  if (browserConfig?.pointerLockControls) {
    releaseControls();
    setStatus("Paused — window lost focus");
  }
});

activateButton.onclick = () => {
  void setActive(!active);
};

resetButton.onclick = () => {
  releaseControls();
  send({type: "reset"});
  setStatus("Reset requested");
};

stopButton.onclick = () => {
  releaseControls();
  send({type: "close"});
  stopped = true;
  setStatus("Stopped");
  activateButton.disabled = true;
  resetButton.disabled = true;
  stopButton.disabled = true;
};

const closePeer = () => {
  releaseControls();
  controls = null;
  if (peer !== null) {
    peer.onconnectionstatechange = null;
    peer.close();
    peer = null;
  }
};

const scheduleReconnect = reason => {
  if (stopped || reconnectTimer !== null) {
    return;
  }
  closePeer();
  if (!browserConfig?.allowReconnect) {
    stopped = true;
    setStatus(reason ? "Disconnected — " + reason : "Disconnected");
    activateButton.disabled = true;
    resetButton.disabled = true;
    stopButton.disabled = true;
    return;
  }

  setStatus(
    reason
      ? "Disconnected — " + reason + "; retrying…"
      : "Disconnected — reconnecting…",
  );
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    void connect();
  }, 1000);
};

async function waitUntilOpen() {
  while (!stopped) {
    const response = await fetch(endpoint("healthz"), {cache: "no-store"});
    if (response.ok && (await response.json()).open) {
      return;
    }
    await new Promise(resolve => window.setTimeout(resolve, 100));
  }
}

async function connect() {
  try {
    await waitUntilOpen();
    if (stopped) {
      return;
    }

    const configResponse = await fetch(endpoint("api/webrtc/config"), {cache: "no-store"});
    if (!configResponse.ok) {
      throw new Error(await configResponse.text());
    }
    browserConfig = await configResponse.json();
    const nextPeer = new RTCPeerConnection({
      iceServers: browserConfig.iceServers,
      iceTransportPolicy: browserConfig.iceTransportPolicy,
    });
    const nextControls = nextPeer.createDataChannel("controls");
    nextPeer.addTransceiver("video", {direction: "recvonly"});
    peer = nextPeer;
    controls = nextControls;

    nextPeer.ontrack = event => {
      video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
    };

    nextControls.onopen = () => {
      setStatus("Ready");
      activateButton.disabled = false;
      resetButton.disabled = false;
      stopButton.disabled = false;
    };
    nextControls.onclose = () => scheduleReconnect("control channel closed");
    nextPeer.onconnectionstatechange = () => {
      if (["failed", "disconnected", "closed"].includes(nextPeer.connectionState)) {
        scheduleReconnect(`WebRTC ${nextPeer.connectionState}`);
      }
    };

    await nextPeer.setLocalDescription(await nextPeer.createOffer());
    const response = await fetch(endpoint("api/webrtc/offer"), {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({
        sdp: nextPeer.localDescription.sdp,
        type: nextPeer.localDescription.type,
        client_id: clientId,
      }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    await nextPeer.setRemoteDescription(await response.json());
  } catch (error) {
    console.error(error);
    scheduleReconnect(error instanceof Error ? error.message : String(error));
  }
}

void connect();
