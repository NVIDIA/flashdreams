// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor() {
    this.handlers = new Map();
    this.style = {};
    this.classList = {toggle() {}};
  }
  addEventListener(name, handler) {
    const handlers = this.handlers.get(name) ?? [];
    handlers.push(handler);
    this.handlers.set(name, handlers);
  }
  emit(name, event = {}) {
    for (const handler of this.handlers.get(name) ?? []) handler(event);
  }
  focus() {}
  setPointerCapture() {}
}
const channels = [];
class Peer extends Element {
  constructor() { super(); Peer.instance = this; }
  addTransceiver() {}
  createDataChannel() {
    const channel = new Element();
    channel.readyState = "open";
    channel.sent = [];
    channel.send = payload => channel.sent.push(JSON.parse(payload));
    channels.push(channel);
    return channel;
  }
}
const video = new Element();
video.videoWidth = 400;
video.videoHeight = 200;
let viewport = {left: 10, top: 20, width: 1000, height: 1000};
video.getBoundingClientRect = () => viewport;
const status = new Element();
const document = {
  getElementById: id => id === "video" ? video : status,
  createElement: () => assert.fail("The viewer must not create link overlays"),
  hasFocus: () => true,
};
const opened = [];
const window = new Element();
window.requestAnimationFrame = () => 1;
window.cancelAnimationFrame = () => {};
// noopener may return null on success as well as when a popup is blocked.
window.open = (...args) => { opened.push(args); return null; };
const context = vm.createContext({
  document, window, RTCPeerConnection: Peer, URL, console,
  navigator: {}, fetch: () => new Promise(() => {}),
});
const script = path.join(__dirname, "../flashdreams/runtime_v2/serving/web/app.js");
vm.runInContext(fs.readFileSync(script, "utf8"), context);

const command = url => channels[0].emit("message", {
  data: JSON.stringify({type: "open_url", url}),
});
const url = "https://localhost:48322/";
command(url);
assert.deepEqual(opened, [[url, "_blank", "noopener,noreferrer"]]);
for (const invalid of [
  "javascript:alert(1)", "file:///tmp/test", "data:text/html,test",
  "https://user:secret@example.test/", "https://@example.test/",
  "https:///example.test/", "https:example.test/", "//example.test/",
  "https://example.test/\npath", "https://example.test/\x85path",
  "https://example.test\\@evil.test/", " https://example.test/",
  "https://bad host/", "https://example.test:99999/",
  "https://example.test/" + "x".repeat(4096), "not a URL", "", null, 42,
]) command(invalid);
assert.equal(opened.length, 1);

// Link metadata is no longer interpreted or replayed by the browser.
channels[0].emit("message", {data: JSON.stringify({type: "links", links: [{url}]})});
channels[0].emit("close");
Peer.instance.connectionState = "failed";
Peer.instance.emit("connectionstatechange");
assert.equal(opened.length, 1);

window.emit("keydown", {target: video, code: "Enter", key: "Enter"});
window.emit("keyup", {target: video, code: "Enter", key: "Enter"});
assert.deepEqual(channels[0].sent, [
  {type: "keyboard", key: "Enter", pressed: true},
  {type: "keyboard", key: "Enter", pressed: false},
]);

const pointer = (x, y, extra = {}) => ({
  clientX: x, clientY: y, button: 0, pointerId: 1, pointerType: "mouse",
  preventDefault() {}, ...extra,
});
// The 2:1 video occupies y=270..770 within the square viewport.
video.emit("pointerdown", pointer(510, 520));
video.emit("pointerup", pointer(510, 520));
assert.deepEqual(channels[1].sent, [
  {type: "mouse", action: "button", x: 0.5, y: 0.5, in_content: true, button: 0, pressed: true},
  {type: "mouse", action: "button", x: 0.5, y: 0.5, in_content: true, button: 0, pressed: false},
]);
// Hits are decided by the server; sending pointer input never opens a local tab.
assert.equal(opened.length, 1);
video.emit("pointerdown", pointer(510, 100));
assert.equal(channels[1].sent.at(-1).y, 0);
assert.equal(channels[1].sent.at(-1).in_content, false);
video.emit("pointerup", pointer(1011, 520));
assert.equal(channels[1].sent.at(-1).x, 1);
assert.equal(channels[1].sent.at(-1).in_content, false);

for (const cancel of [
  () => video.emit("pointercancel", pointer(510, 520)),
  () => window.emit("blur"),
]) {
  video.emit("pointerdown", pointer(510, 520));
  cancel();
  assert.equal(channels[1].sent.at(-1).pressed, false);
  assert.equal(channels[1].sent.at(-1).cancelled, true);
  const count = channels[1].sent.length;
  video.emit("pointerup", pointer(510, 520));
  assert.equal(channels[1].sent.length, count);
}

viewport = {left: 10, top: 20, width: 1000, height: 500};
video.emit("pointerdown", pointer(510, 270));
assert.equal(channels[1].sent.at(-1).y, 0.5);
video.emit("pointerup", pointer(510, 270));

const mouseCount = channels[1].sent.length;
for (const event of ["pointermove", "pointerdown", "pointerup", "pointercancel"]) {
  video.emit(event, pointer(510, 270, {pointerType: "touch"}));
}
assert.equal(channels[1].sent.length, mouseCount, "Touch must not also become a mouse click");
const touch = {identifier: 7, clientX: 510, clientY: 270, force: 0.5};
video.emit("touchstart", {touches: [touch], changedTouches: [touch], preventDefault() {}});
video.emit("touchend", {touches: [], changedTouches: [touch], preventDefault() {}});
assert.deepEqual(channels[0].sent.slice(-2), [
  {type: "touch", action: "start", touch_id: 7, x: 0.5, y: 0.5, in_content: true, pressure: 0.5, primary: true},
  {type: "touch", action: "end", touch_id: 7, x: 0.5, y: 0.5, in_content: true, pressure: 0.5, primary: false},
]);
