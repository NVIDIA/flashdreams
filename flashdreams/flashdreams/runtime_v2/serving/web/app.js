// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const peer = new RTCPeerConnection();
const controls = peer.createDataChannel("controls");
peer.addTransceiver("video", {direction: "recvonly"});

peer.ontrack = event => {
  document.getElementById("video").srcObject =
    event.streams[0] ?? new MediaStream([event.track]);
};

const send = payload => {
  if (controls.readyState === "open") {
    controls.send(JSON.stringify(payload));
  }
};

window.addEventListener("keydown", event => {
  send({type: "keyboard", key: event.key, pressed: true});
});

window.addEventListener("keyup", event => {
  send({type: "keyboard", key: event.key, pressed: false});
});

let activationPressed = false;
document.getElementById("activate").onclick = event => {
  activationPressed = !activationPressed;
  event.currentTarget.textContent =
    activationPressed ? "Deactivate" : "Activate";
  send({type: "keyboard", key: "r", pressed: activationPressed});
};

document.getElementById("reset").onclick = () => {
  send({type: "reset"});
};

window.addEventListener("beforeunload", () => send({type: "close"}));

async function connect() {
  while (true) {
    const health = await fetch("/healthz");
    if (health.ok && (await health.json()).open) {
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  await peer.setLocalDescription(await peer.createOffer());
  const response = await fetch("/api/webrtc/offer", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(peer.localDescription),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  await peer.setRemoteDescription(await response.json());
}

connect();
