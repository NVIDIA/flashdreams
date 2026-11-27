// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function fixture({ status = 200 } = {}) {
  class Element {
    constructor() {
      this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {};
      this.classList = { toggle() {}, add() {}, remove() {} };
      this.textContent = ''; this.value = ''; this.children = [];
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    removeAttribute(name) { delete this.attributes[name]; }
    play() { return Promise.resolve(); }
  }
  const elements = new Map();
  const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
  const gearItems = [...html.matchAll(/<button[^>]+id="gear-([prnd])"[^>]+data-gear="([PRND])"/g)].map(match => {
    const element = new Element(); element.setAttribute('data-gear', match[2]);
    elements.set(`gear-${match[1]}`, element); return element;
  });
  const document = {
    hidden: false, hasFocus: () => true,
    documentElement: new Element(), listeners: {},
    createElement: () => new Element(),
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    querySelectorAll: selector => selector === '.gear-item' ? gearItems : [],
    addEventListener(name, handler) { this.listeners[name] = handler; },
  };
  const messages = [], requests = [], peers = [];
  class Peer {
    constructor() {
      this.iceGatheringState = 'new'; this.connectionState = 'new';
      this.listeners = {}; peers.push(this);
    }
    createDataChannel() {
      this.channel = { readyState: 'connecting', send: message => messages.push(JSON.parse(message)) };
      return this.channel;
    }
    addTransceiver() {}
    async createOffer() { return { type: 'offer', sdp: 'before-ice' }; }
    async setLocalDescription(offer) {
      this.localDescription = offer;
      setTimeout(() => {
        this.localDescription = { type: 'offer', sdp: 'with-ice-candidates' };
        this.iceGatheringState = 'complete';
        this.listeners.icegatheringstatechange?.();
      }, 0);
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    removeEventListener(name) { delete this.listeners[name]; }
    async setRemoteDescription(answer) { this.remoteDescription = answer; }
    close() { this.channel.readyState = 'closed'; }
  }
  const window = { listeners: {}, location: { hash: '' }, setInterval: () => 1,
    addEventListener(name, handler) { this.listeners[name] = handler; } };
  const source = fs.readFileSync(path.join(__dirname, '../src/components/CockpitApp.ts'), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  vm.runInNewContext(code, {
    exports, document, window, navigator: { getGamepads: () => [] },
    RTCPeerConnection: Peer, setTimeout, clearTimeout, clearInterval() {},
    URL, URLSearchParams, AbortSignal, console, performance,
    localStorage: { getItem: () => null, setItem() {} },
    fetch: async (url, options) => {
      requests.push({ url, body: JSON.parse(options.body) });
      return { ok: status === 200, status, json: async () => ({ type: 'answer', sdp: 'answer' }), text: async () => 'backend unavailable' };
    },
  });
  const app = new exports.CockpitApp();
  const ready = async () => {
    for (let i = 0; i < 100 && !peers.at(-1).remoteDescription && !document.getElementById('connection-status').textContent.includes('连接失败'); i++) {
      await new Promise(resolve => setTimeout(resolve, 2));
    }
    return peers.at(-1);
  };
  const key = (name, pressed, shiftKey = false) => window.listeners[pressed ? 'keydown' : 'keyup']({ key: name, shiftKey, preventDefault() {}, target: null });
  const cards = () => document.getElementById('camera-list').children;
  const receive = (peer, data) => peer.channel.onmessage({ data: JSON.stringify(data) });
  return { app, document, window, messages, requests, peers, ready, key, cards, receive, gearItems };
}

test('ICE offer, camera selection, D steering, release, reverse and real telemetry', async () => {
  const f = fixture();
  try {
    const peer = await f.ready();
    assert.equal(f.requests[0].url, '/api/webrtc/offer');
    assert.equal(f.requests[0].body.sdp, 'with-ice-candidates');
    peer.channel.readyState = 'open'; peer.channel.onopen();
    assert.ok(!f.messages.some(message => message.type === 'select_camera'));
    f.key('d', true); f.key('w', true);
    assert.equal(f.messages.at(-1).action, 'state');
    assert.equal(f.messages.at(-1).steering, 1); // backend converts right to negative yaw
    assert.equal(f.messages.at(-1).throttle, 1);
    assert.equal(f.document.getElementById('speed-num').textContent, '—');
    f.key('d', false);
    assert.equal(f.messages.at(-1).steering, 0);
    const before = f.messages.length;
    peer.channel.onmessage({ data: JSON.stringify({ type: 'telemetry', speed_kmh: 12.4, steer: 1, throttle: 0 }) });
    assert.equal(f.document.getElementById('speed-num').textContent, '12');
    assert.equal(f.messages.length, before); // no command feedback from old telemetry
    f.app.setGear('R');
    assert.equal(f.messages.at(-1).reverse, true);
    f.window.listeners.blur();
    const neutral = f.messages.at(-2);
    assert.equal(neutral.throttle, 0); assert.equal(neutral.steering, 0);
    assert.equal(f.messages.at(-1).focused, false);
    assert.ok(!f.messages.some(message => message.type === 'keyboard')); // one authoritative control path
    peer.channel.onmessage({ data: JSON.stringify({ type: 'error', message: 'invalid control' }) });
    assert.match(f.document.getElementById('connection-status').textContent, /invalid control/);
    f.document.getElementById('setting-backend-url').value = 'http://another-host:8089/';
    f.document.getElementById('btn-reconnect-server').listeners.click();
    await f.ready();
    assert.equal(f.requests.at(-1).url, 'http://another-host:8089/api/webrtc/offer');
  } finally { f.app.destroy(); }
});

test('HTTP failure is visible instead of pretending the simulation works', async () => {
  const f = fixture({ status: 502 });
  try {
    await f.ready();
    assert.match(f.document.getElementById('connection-status').textContent, /连接失败.*502/);
    assert.equal(f.document.getElementById('speed-num').textContent, '—');
  } finally { f.app.destroy(); }
});

test('recording commands wait for backend acknowledgement and expose the finalized archive', async () => {
  const f = fixture();
  try {
    const peer = await f.ready(); peer.channel.readyState = 'open'; peer.channel.onopen();
    f.document.getElementById('record-session-start').listeners.click();
    assert.deepEqual(f.messages.at(-1), { type: 'recording', action: 'start' });
    peer.channel.onmessage({ data: JSON.stringify({ type: 'recording', state: 'recording', id: 'drive-test', frames: 12 }) });
    assert.equal(f.document.getElementById('record-session-start').disabled, true);
    assert.match(f.document.getElementById('record-session-status').textContent, /12 帧/);
    f.document.getElementById('record-session-stop').listeners.click();
    assert.deepEqual(f.messages.at(-1), { type: 'recording', action: 'stop' });
    peer.channel.onmessage({ data: JSON.stringify({ type: 'recording', state: 'ready', id: 'drive-test', frames: 12 }) });
    const link = f.document.getElementById('record-session-download');
    assert.equal(link.hidden, false);
    assert.match(link.href, /^\/api\/drive-exports\/drive-test.zip\?token=/);
  } finally { f.app.destroy(); }
});


test('all four gear buttons and Shift shortcuts', async () => {
  const f = fixture();
  try {
    const peer = await f.ready(); peer.channel.readyState = 'open'; peer.channel.onopen();
    assert.equal(f.gearItems.length, 4);
    for (const item of f.gearItems) {
      item.listeners.click();
      const gear = item.getAttribute('data-gear');
      assert.equal(f.messages.at(-1).gear, gear);
      assert.equal(f.messages.at(-1).brake, gear === 'P' ? 1 : 0);
      peer.channel.onmessage({ data: JSON.stringify({ type: 'telemetry', speed_kmh: 0, gear }) });
      assert.equal(f.app.telemetry.gear, gear);
    }
    for (const gear of ['N', 'R', 'D', 'P']) {
      f.key(gear.toLowerCase(), true, true);
      assert.equal(f.messages.at(-1).gear, gear);
    }
  } finally { f.app.destroy(); }
});

test('camera cards and selection follow backend count, order and reconnects', async () => {
  const f = fixture();
  try {
    const peer = await f.ready();
    peer.channel.readyState = 'open'; peer.channel.onopen();
    assert.equal(f.cards().length, 0);
    f.app.cycleCamera(1);
    assert.ok(!f.messages.some(message => message.type === 'select_camera'));

    const names = ['camera_cross_left_120fov', 'camera_front_wide_120fov',
      'camera_rear_tele_30fov', 'camera_cross_right_120fov', 'camera_rear_right_70fov',
      'camera_rear_left_70fov', 'custom_camera'];
    for (const count of [1, 4, 5, 6, 7]) {
      f.receive(peer, { type: 'telemetry', active_view: names[Math.min(count - 1, 1)], camera_names: names.slice(0, count) });
      assert.equal(f.document.getElementById('camera-count').textContent, `${count} 个视角`);
      assert.deepEqual(f.cards().map(card => card.dataset.camera), names.slice(0, count));
    }
    assert.equal(f.cards()[1].getAttribute('aria-pressed'), 'true');
    assert.equal(f.cards()[6].getAttribute('aria-label'), '切换到custom_camera');
    const image = 'data:image/jpeg;base64,AA==';
    f.receive(peer, { type: 'camera_preview', camera: names[6], image });
    f.receive(peer, { type: 'camera_preview', camera: 'disabled_camera', image });
    assert.equal(f.cards().length, 7);
    assert.equal(f.cards()[6].children[0].children[0].src, image);
    assert.equal(f.cards()[6].children[0].children[1].hidden, true);
    f.cards()[6].listeners.click();
    assert.deepEqual(f.messages.at(-1), { type: 'select_camera', camera: names[6], view_index: 6 });
    f.app.cycleCamera(1);
    assert.deepEqual(f.messages.at(-1), { type: 'select_camera', camera: names[0], view_index: 0 });
    f.app.cycleCamera(-1);
    assert.equal(f.messages.at(-1).camera, names[6]);
    f.receive(peer, { type: 'telemetry', active_view: names[1], camera_names: [names[1]] });
    assert.equal(f.cards().length, 1);
    assert.equal(f.cards()[0].dataset.camera, names[1]);
    const messageCount = f.messages.length;
    f.app.cycleCamera(1);
    assert.equal(f.messages.length, messageCount);

    f.document.getElementById('btn-reconnect-server').listeners.click();
    const nextPeer = await f.ready();
    assert.equal(f.cards().length, 0);
    f.receive(peer, { type: 'camera_preview', camera: names[6], image });
    assert.equal(f.cards().length, 0);
    // A running backend predating camera_names still sends ordered previews.
    f.receive(nextPeer, { type: 'telemetry', active_view: names[2] });
    for (const name of [names[2], names[0], names[2]]) {
      f.receive(nextPeer, { type: 'camera_preview', camera: name, image });
    }
    assert.deepEqual(f.cards().map(card => card.dataset.camera), [names[2], names[0]]);
    assert.equal(f.document.getElementById('camera-count').textContent, '2 个视角');
    assert.equal(f.cards()[0].getAttribute('aria-pressed'), 'true');
  } finally { f.app.destroy(); }
});
