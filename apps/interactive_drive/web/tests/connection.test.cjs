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
      this.textContent = ''; this.value = '';
    }
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
  const cards = [...html.matchAll(/<button[^>]+class="camera-card[^"]*"[^>]+data-channel="(\d+)"/g)].map(match => {
    const element = new Element(); element.setAttribute('data-channel', match[1]); return element;
  });
  const document = {
    hidden: false, hasFocus: () => true,
    documentElement: new Element(), listeners: {},
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    querySelectorAll: selector => selector === '.gear-item' ? gearItems : selector === '.camera-card' ? cards : [],
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
  const window = { listeners: {}, setInterval: () => 1,
    addEventListener(name, handler) { this.listeners[name] = handler; } };
  const source = fs.readFileSync(path.join(__dirname, '../src/components/CockpitApp.ts'), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  vm.runInNewContext(code, {
    exports, document, window, navigator: { getGamepads: () => [] },
    RTCPeerConnection: Peer, setTimeout, clearTimeout, clearInterval() {},
    URL, AbortSignal, console, performance,
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
  return { app, document, window, messages, requests, peers, ready, key, cards, gearItems };
}

test('ICE offer, camera selection, D steering, release, reverse and real telemetry', async () => {
  const f = fixture();
  try {
    const peer = await f.ready();
    assert.equal(f.requests[0].url, '/api/webrtc/offer');
    assert.equal(f.requests[0].body.sdp, 'with-ice-candidates');
    peer.channel.readyState = 'open'; peer.channel.onopen();
    assert.equal(f.messages[0].type, 'select_camera');
    assert.equal(f.messages[0].camera, 'camera_front_wide_120fov');
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


test('all four gear buttons, Shift shortcuts, seven preview images and camera selection', async () => {
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
    assert.equal(f.cards.length, 7);
    const names = ['front_wide_120fov', 'cross_right_120fov', 'rear_right_70fov', 'rear_tele_30fov', 'rear_left_70fov', 'cross_left_120fov', 'front_tele_30fov'];
    names.forEach((name, i) => {
      const image = `data:image/jpeg;base64,${i}`;
      peer.channel.onmessage({ data: JSON.stringify({ type: 'camera_preview', camera: `camera_${name}`, image }) });
      assert.equal(f.document.getElementById(`camera-preview-${i}`).src, image);
      assert.equal(f.document.getElementById(`camera-preview-${i}`).hidden, false);
      assert.equal(f.document.getElementById(`camera-waiting-${i}`).hidden, true);
    });
    f.cards[6].listeners.click();
    assert.equal(f.messages.at(-1).camera, 'camera_front_tele_30fov');
    assert.equal(f.messages.at(-1).view_index, 6);
    assert.ok(!f.messages.some(message => message.camera === 'all'));
    f.document.getElementById('btn-reconnect-server').listeners.click();
    await f.ready();
    assert.equal(f.document.getElementById('camera-preview-0').hidden, true);
  } finally { f.app.destroy(); }
});
