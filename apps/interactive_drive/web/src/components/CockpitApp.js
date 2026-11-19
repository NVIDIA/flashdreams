/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export class CockpitApp {
  constructor() {
    this.peer = null;
    this.controlChannel = null;
    this.videoEl = document.getElementById('video');

    this.pressedKeys = new Set();
    this.currentSteer = 0;
    this.currentThrottle = 0;
    this.currentBrake = 0;
    this.currentGear = 'D';
    this.currentSpeedKmh = 33;
    this.warningActive = true;

    this.telemetry = {
      timestamp_us: 0,
      step_index: 0,
      speed_mps: 9.16,
      speed_kmh: 33,
      gear: 'D',
      steer: 0,
      steer_deg: 0,
      throttle: 0,
      brake: 0,
      handbrake: false,
      ego_pose: { x: 0, y: 0, z: 0, heading_deg: 0 },
      model_latency_ms: 28.5,
      render_fps: 30,
      denoise_steps: 4,
      blocks_generated: 12,
      total_views: 7,
      active_view: 'camera_front_wide_120fov',
      warning: '车辆即将断电，请安全靠边停车',
      warning_detail: '车辆异常，联系线上服务中心以获得支持',
    };

    this.initKeyboard();
    this.initWebRTC();
    this.startControlLoop();
    this.bindUI();
  }

  initKeyboard() {
    window.addEventListener('keydown', (e) => {
      const key = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', ' ', 'r'].includes(key)) {
        e.preventDefault();
      }
      this.pressedKeys.add(key);
      if (key === 'r') {
        this.resetSimulation();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.pressedKeys.delete(e.key.toLowerCase());
    });
  }

  startControlLoop() {
    window.setInterval(() => {
      // Smooth steering ramp & return
      const targetSteer = (this.pressedKeys.has('d') ? 1 : 0) - (this.pressedKeys.has('a') ? 1 : 0);
      this.currentSteer += (targetSteer - this.currentSteer) * 0.25;

      // Throttle & Brake
      const targetThrottle = this.pressedKeys.has('w') ? 1 : 0;
      this.currentThrottle += (targetThrottle - this.currentThrottle) * 0.3;

      const targetBrake = (this.pressedKeys.has('s') || this.pressedKeys.has(' ')) ? 1 : 0;
      this.currentBrake += (targetBrake - this.currentBrake) * 0.4;

      // Simulated speed change when not connected to live backend
      if (this.currentThrottle > 0) {
        this.currentSpeedKmh = Math.min(120, this.currentSpeedKmh + this.currentThrottle * 1.2);
      } else if (this.currentBrake > 0) {
        this.currentSpeedKmh = Math.max(0, this.currentSpeedKmh - this.currentBrake * 2.5);
      } else {
        this.currentSpeedKmh = Math.max(0, this.currentSpeedKmh - 0.2);
      }

      this.currentGear = this.currentSpeedKmh === 0 && this.pressedKeys.has('s') ? 'R' : 'D';

      this.sendDriverCommand();
      this.renderTelemetry();
    }, 33);
  }

  sendDriverCommand() {
    if (!this.controlChannel || this.controlChannel.readyState !== 'open') return;

    const cmd = {
      type: 'driver_command',
      timestamp_us: performance.now() * 1000,
      steer: this.currentSteer,
      throttle: this.currentThrottle,
      brake: this.currentBrake,
      reverse: this.currentGear === 'R',
    };

    try {
      this.controlChannel.send(JSON.stringify(cmd));
    } catch {
      // Ignored
    }
  }

  resetSimulation() {
    if (!this.controlChannel || this.controlChannel.readyState !== 'open') return;
    this.controlChannel.send(JSON.stringify({ type: 'reset_simulation' }));
  }

  async initWebRTC() {
    try {
      this.peer = new RTCPeerConnection();
      this.controlChannel = this.peer.createDataChannel('controls');

      this.controlChannel.onopen = () => {
        const statusEl = document.getElementById('conn-status');
        if (statusEl) statusEl.textContent = 'SIM LIVE';
      };

      this.controlChannel.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'telemetry') {
            this.telemetry = { ...this.telemetry, ...data };
            this.currentSpeedKmh = this.telemetry.speed_kmh;
            this.currentGear = this.telemetry.gear;
            if (this.telemetry.warning) {
              this.warningActive = true;
            }
          }
        } catch {
          // Ignored
        }
      };

      this.peer.addTransceiver('video', { direction: 'recvonly' });

      this.peer.ontrack = (event) => {
        if (this.videoEl) {
          this.videoEl.srcObject = event.streams[0] || new MediaStream([event.track]);
          this.videoEl.play().catch(() => {});
        }
      };

      const offer = await this.peer.createOffer();
      await this.peer.setLocalDescription(offer);

      const resp = await fetch('/api/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
      });

      if (resp.ok) {
        const answer = await resp.json();
        await this.peer.setRemoteDescription(new RTCSessionDescription(answer));
      }
    } catch (err) {
      console.warn('[WebRTC] Backend not connected yet. Operating in local interactive simulation preview mode.', err);
    }
  }

  bindUI() {
    const hotlineBtn = document.getElementById('alert-action-btn');
    if (hotlineBtn) {
      hotlineBtn.addEventListener('click', () => {
        const card = document.getElementById('alert-card');
        if (card) {
          card.classList.add('entering');
          setTimeout(() => {
            card.style.display = 'none';
          }, 240);
        }
      });
    }

    const cards = document.querySelectorAll('.camera-card');
    cards.forEach((card, idx) => {
      card.addEventListener('click', () => {
        cards.forEach((c) => c.classList.remove('active'));
        card.classList.add('active');
        const viewNames = ['前视 (主摄 120°)', '后重视角', '左侧环视', '右侧环视', 'World Scenario (鹰眼)'];
        const tag = document.getElementById('view-tag-label');
        if (tag) tag.textContent = `< ${viewNames[idx] || '前视'}`;

        if (this.controlChannel && this.controlChannel.readyState === 'open') {
          this.controlChannel.send(JSON.stringify({ type: 'select_camera', view_index: idx }));
        }
      });
    });
  }

  renderTelemetry() {
    const speedEl = document.getElementById('speed-num');
    if (speedEl) speedEl.textContent = Math.round(this.currentSpeedKmh).toString();

    ['p', 'r', 'n', 'd'].forEach((g) => {
      const el = document.getElementById(`gear-${g}`);
      if (el) el.classList.toggle('active', this.currentGear.toLowerCase() === g);
    });

    const steerDeg = this.currentSteer * 45;
    const wheel = document.getElementById('steering-wheel');
    if (wheel) {
      wheel.style.transform = `rotate(${steerDeg}deg)`;
    }
    const steerAngleText = document.getElementById('steer-angle-val');
    if (steerAngleText) {
      steerAngleText.textContent = `${this.currentSteer > 0 ? '+' : ''}${(this.currentSteer * 270).toFixed(1)}°`;
    }

    const throttleBar = document.getElementById('throttle-fill');
    if (throttleBar) throttleBar.style.width = `${Math.round(this.currentThrottle * 100)}%`;
    const throttleVal = document.getElementById('throttle-val');
    if (throttleVal) throttleVal.textContent = `${Math.round(this.currentThrottle * 100)}%`;

    const brakeBar = document.getElementById('brake-fill');
    if (brakeBar) brakeBar.style.width = `${Math.round(this.currentBrake * 100)}%`;
    const brakeVal = document.getElementById('brake-val');
    if (brakeVal) brakeVal.textContent = `${Math.round(this.currentBrake * 100)}%`;

    this.renderMinimap();
  }

  renderMinimap() {
    const canvas = document.getElementById('minimap-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = (canvas.width = canvas.clientWidth);
    const h = (canvas.height = canvas.clientHeight);

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Lane lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w * 0.25, 0);
    ctx.lineTo(w * 0.25, h);
    ctx.moveTo(w * 0.75, 0);
    ctx.lineTo(w * 0.75, h);
    ctx.stroke();

    // Divider
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = 'rgba(255, 214, 10, 0.6)';
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.5, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Ego vehicle
    const cx = w * 0.5;
    const cy = h * 0.6;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate((this.currentSteer * Math.PI) / 6);

    ctx.fillStyle = '#007aff';
    ctx.shadowColor = 'rgba(0, 122, 255, 0.5)';
    ctx.shadowBlur = 10;
    ctx.fillRect(-12, -22, 24, 44);

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(0, -28);
    ctx.lineTo(-6, -20);
    ctx.lineTo(6, -20);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new CockpitApp();
});
