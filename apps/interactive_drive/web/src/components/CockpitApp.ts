/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SimulationTelemetry, Gear } from '../protocol/types';

const cameraLabels = new Map([
  ['camera_front_wide_120fov', '前向广角 120°'],
  ['camera_cross_right_120fov', '右前侧向 120°'],
  ['camera_rear_right_70fov', '右后侧向 70°'],
  ['camera_rear_tele_30fov', '正后远焦 30°'],
  ['camera_rear_left_70fov', '左后侧向 70°'],
  ['camera_cross_left_120fov', '左前侧向 120°'],
  ['camera_front_tele_30fov', '正前远焦 30°'],
]);

/**
 * High-fidelity automotive haptic engine
 * Accurately models ABS hydraulic pulsing, crisp mechanical shifting, and launch torque
 * Eliminates cheap continuous plastic motor rattling
 */
class HapticEngine {
  private enabled = true;
  private lastRumbleTime = 0;
  private absPulseTimer = 0;
  private lastThrottle = 0;

  setEnabled(val: boolean): void {
    this.enabled = val;
  }

  private play(durationMs: number, strong: number, weak: number): void {
    if (!this.enabled) return;
    const now = performance.now();
    // Enforce 24ms minimum motor cadence to prevent overlapping buzzing spam
    if (now - this.lastRumbleTime < 24) return;
    this.lastRumbleTime = now;

    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (let i = 0; i < gamepads.length; i++) {
      const gp = gamepads[i];
      if (gp && gp.connected && (gp as any).vibrationActuator) {
        try {
          (gp as any).vibrationActuator.playEffect('dual-rumble', {
            startDelay: 0,
            duration: Math.round(durationMs),
            weakMagnitude: Math.min(1.0, Math.max(0, weak)),
            strongMagnitude: Math.min(1.0, Math.max(0, strong)),
          });
        } catch {
          // Unsupported
        }
      }
    }
  }

  /**
   * Crisp 22ms mechanical shift notch (清脆机械挂挡段落感，绝无塑料蜂鸣)
   */
  gearShift(): void {
    this.play(22, 0.42, 0.15);
  }

  /**
   * Scene reset transient (重置场景瞬时阻尼)
   */
  resetImpact(): void {
    this.play(70, 0.6, 0.35);
  }

  /**
   * Evaluated every 33ms in control loop
   * Disables vibration during gentle driving, and fires 14Hz pulses on heavy ABS lockup
   */
  update(brake: number, throttle: number, speedKmh: number, handbrake: boolean): void {
    if (!this.enabled) return;
    const now = performance.now();

    // 1. ABS & Hard Braking: 14Hz intermittent hydraulic pulsing (14Hz 节奏点刹脉冲)
    // Gentle braking (0 - 80%) is completely quiet and smooth, just like real cars!
    const isHardBraking = handbrake || brake > 0.82;
    if (isHardBraking && speedKmh > 2.0) {
      if (now - this.absPulseTimer >= 70) {
        this.absPulseTimer = now;
        // 24ms firm hit on left grip followed by 46ms silence = distinct ABS pedal kick
        const strength = handbrake ? 0.65 : 0.35 + (brake - 0.82) * 1.5;
        this.play(24, Math.min(0.68, strength), 0.15);
      }
      return;
    }

    // 2. Launch Wheel Spin (起步大油门弹射微震)
    // Only triggers on standing start (< 16 km/h) with heavy throttle (> 88%), NEVER during cruising!
    const isLaunch = throttle > 0.88 && speedKmh < 16.0;
    const isKickdown = throttle - this.lastThrottle > 0.65;
    this.lastThrottle = throttle;

    if (isLaunch || isKickdown) {
      if (now - this.absPulseTimer >= 90) {
        this.absPulseTimer = now;
        // Subtle high-frequency texture on right grip, zero heavy motor noise
        this.play(30, 0.0, 0.28);
      }
      return;
    }

    // Cruising / gentle braking: solid, silent, zero vibration noise!
  }
}

export class CockpitApp {
  private peer: RTCPeerConnection | null = null;
  private controlChannel: RTCDataChannel | null = null;
  private videoEl: HTMLVideoElement;

  // Vehicle Control & Gear State
  private pressedKeys = new Set<string>();
  private currentSteer = 0;       // [-1, 1]
  private currentThrottle = 0;    // [0, 1]
  private currentBrake = 0;       // [0, 1]
  private currentGear: Gear = 'D';
  private currentSpeedKmh = 0;
  private warningActive = false;  // Disabled by default; only shown when real warning occurs
  private backendUrl = '';
  private selectedCamera: string | undefined;
  private lastCommand = '';
  private hasTelemetry = false;
  private clockInterval: number | null = null;
  private haptics = new HapticEngine();

  private readonly cameraCards = new Map<string, {
    button: HTMLButtonElement;
    image: HTMLImageElement;
    waiting: HTMLSpanElement;
  }>();

  private telemetry: Partial<SimulationTelemetry> = {};

  private loopInterval: number | null = null;
  private lastGamepadReset = false;
  private lastGamepadButtons = new Set<number>();

  public destroy(): void {
    if (this.loopInterval !== null) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }
    this.releaseControls();
    if (this.clockInterval !== null) clearInterval(this.clockInterval);
    if (this.peer) {
      this.peer.close();
      this.peer = null;
    }
  }

  constructor() {
    this.videoEl = document.getElementById('video') as HTMLVideoElement;
    this.initTheme();
    this.startClock();
    this.initKeyboard();
    this.initWebRTC();
    this.startControlLoop();
    this.bindUI();
    this.bindRecording();
    this.bindSettings();
  }

  private startClock(): void {
    const update = () => {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, '0');
      const m = String(now.getMinutes()).padStart(2, '0');
      const clockEl = document.getElementById('top-clock');
      if (clockEl) clockEl.textContent = `${h}:${m}`;
    };
    update();
    this.clockInterval = window.setInterval(update, 1000);
  }

  private initTheme(): void {
    const saved = localStorage.getItem('cockpit_theme') || 'dark';
    this.setTheme(saved as 'dark' | 'light');

    document.getElementById('theme-toggle-btn')?.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      this.setTheme(current === 'dark' ? 'light' : 'dark');
    });

    document.getElementById('btn-theme-dark')?.addEventListener('click', () => this.setTheme('dark'));
    document.getElementById('btn-theme-light')?.addEventListener('click', () => this.setTheme('light'));
  }

  private setTheme(theme: 'dark' | 'light'): void {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cockpit_theme', theme);
    const sunIcon = document.getElementById('theme-icon-sun');
    const moonIcon = document.getElementById('theme-icon-moon');
    if (sunIcon && moonIcon) {
      sunIcon.style.display = theme === 'dark' ? 'block' : 'none';
      moonIcon.style.display = theme === 'light' ? 'block' : 'none';
    }
  }

  /**
   * Shift gear cleanly (P / R / N / D)
   */
  public setGear(gear: Gear): void {
    if (this.currentGear !== gear) {
      this.haptics.gearShift();
    }
    this.currentGear = gear;

    // Top bar (P) icon
    const pIcon = document.getElementById('top-p-icon');
    if (pIcon) {
      pIcon.classList.toggle('active', gear === 'P');
    }

    // Right rail gear selector buttons
    ['p', 'r', 'n', 'd'].forEach((g) => {
      const el = document.getElementById(`gear-${g}`);
      if (el) el.classList.toggle('active', gear.toLowerCase() === g);
    });

    this.updateControls();
  }

  private sendEvent(payload: object): boolean {
    if (!this.controlChannel || this.controlChannel.readyState !== 'open') return false;
    try {
      this.controlChannel.send(JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }

  private bindRecording(): void {
    for (const action of ['start', 'stop'] as const) {
      document.getElementById(`record-session-${action}`)?.addEventListener('click', () => {
        const status = document.getElementById('record-session-status');
        if (!this.sendEvent({ type: 'recording', action })) {
          if (status) status.textContent = '请先连接模型后端';
          return;
        }
        if (status) status.textContent = action === 'start' ? '等待下一个生成块开始录制…' : '等待当前生成块完成并归档…';
      });
    }
  }

  private renderRecording(data: { state: string; id?: string; frames?: number; message?: string }): void {
    const start = document.getElementById('record-session-start') as HTMLButtonElement | null;
    const stop = document.getElementById('record-session-stop') as HTMLButtonElement | null;
    const download = document.getElementById('record-session-download') as HTMLAnchorElement | null;
    const status = document.getElementById('record-session-status');
    if (start) start.disabled = data.state === 'recording';
    if (stop) stop.disabled = data.state !== 'recording';
    if (status) status.textContent = data.state === 'recording' ? `正在录制 · ${data.frames ?? 0} 帧` : data.state === 'ready' ? `已归档 · ${data.frames ?? 0} 帧` : data.state === 'error' ? `录制失败：${data.message || '没有可导出的帧'}` : '未录制';
    if (download) {
      download.hidden = data.state !== 'ready';
      if (data.state === 'ready' && data.id && /^[a-z0-9-]+$/.test(data.id)) {
        const token = document.getElementById('drive-export-token')?.getAttribute('content') || '';
        download.href = `/api/drive-exports/${data.id}.zip?token=${encodeURIComponent(token)}`;
      }
    }
  }

  public resetSimulation(): void {
    this.haptics.resetImpact();
    this.sendEvent({ type: 'keyboard', key: 'r', pressed: true });
    setTimeout(() => {
      this.sendEvent({ type: 'keyboard', key: 'r', pressed: false });
    }, 100);
  }

  private initKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, [contenteditable="true"]')) return;
      const k = e.key.toLowerCase();
      if (e.shiftKey && ['p', 'r', 'n', 'd'].includes(k)) {
        e.preventDefault();
        if (!e.repeat) this.setGear(k.toUpperCase() as Gear);
        return;
      }
      if (['w', 'a', 's', 'd', ' '].includes(k)) {
        e.preventDefault();
        this.pressedKeys.add(k);
        this.updateControls();
      } else if (!e.repeat) {
        if (k === 'p') this.setGear('P');
        if (k === 'n') this.setGear('N');
        if (k === 'b') this.setGear('R');
        if (k === 'r') this.resetSimulation();
      }
    });
    window.addEventListener('keyup', (e) => {
      if (this.pressedKeys.delete(e.key.toLowerCase())) this.updateControls();
    });
    window.addEventListener('blur', () => this.releaseControls());
    window.addEventListener('pagehide', () => this.releaseControls());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.releaseControls();
    });
  }

  private releaseControls(): void {
    this.pressedKeys.clear();
    this.currentSteer = this.currentThrottle = this.currentBrake = 0;
    this.sendDriverCommand();
    this.sendEvent({ type: 'focus', focused: false });
    this.lastCommand = ''; // Focus loss clears the backend controller; resend after returning.
    this.renderTelemetry();
  }

  /**
   * Gamepad API support (Xbox, PS4/PS5, Standard controllers)
   */
  private pollGamepad(): { steer: number; throttle: number; brake: number; reset: boolean } | null {
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    let gp: Gamepad | null = null;
    for (let i = 0; i < gamepads.length; i++) {
      if (gamepads[i] && gamepads[i]!.connected) {
        gp = gamepads[i];
        break;
      }
    }

    if (!gp) return null;

    // Triggers: RT (Button 7) = Throttle, LT (Button 6) = Brake
    const rawRT = typeof gp.buttons[7] === 'object' ? gp.buttons[7].value : (gp.buttons[7] ? 1 : 0);
    const rawLT = typeof gp.buttons[6] === 'object' ? gp.buttons[6].value : (gp.buttons[6] ? 1 : 0);

    // Tight 0.02 deadzone with linear scaling (removes empty travel / 消除虚位)
    const throttle = rawRT > 0.02 ? (rawRT - 0.02) / 0.98 : 0;
    const brake = rawLT > 0.02 ? (rawLT - 0.02) / 0.98 : 0;

    // Steering: Left stick X with tight 0.025 deadzone + progressive curve
    const rawSteer = gp.axes[0] || 0;
    let steer = 0;
    if (Math.abs(rawSteer) > 0.025) {
      const sign = Math.sign(rawSteer);
      const norm = (Math.abs(rawSteer) - 0.025) / 0.975;
      steer = sign * (0.35 * norm + 0.65 * Math.pow(norm, 2.2));
    }

    // Forza Horizon standard buttons mapping:
    // A (0): Handbrake (手刹急刹)
    const handbrake = Boolean(gp.buttons[0]?.pressed);

    // Y (3): Reset Simulation (Forza Horizon Rewind/Reset)
    const reset = Boolean(gp.buttons[3]?.pressed);

    // LB (4) / RB (5): Camera View Previous / Next
    if (gp.buttons[4]?.pressed && !this.lastGamepadButtons.has(4)) {
      this.cycleCamera(-1);
    }
    if (gp.buttons[5]?.pressed && !this.lastGamepadButtons.has(5)) {
      this.cycleCamera(1);
    }

    // B (1) / X (2): Shift Up / Down
    if (gp.buttons[1]?.pressed && !this.lastGamepadButtons.has(1)) {
      this.shiftStep(1); // Shift up
    }
    if (gp.buttons[2]?.pressed && !this.lastGamepadButtons.has(2)) {
      this.shiftStep(-1); // Shift down
    }

    // D-Pad: Direct gear shift (Up: D, Down: R, Left: P, Right: N)
    if (gp.buttons[12]?.pressed) this.setGear('D');
    if (gp.buttons[13]?.pressed) this.setGear('R');
    if (gp.buttons[14]?.pressed) this.setGear('P');
    if (gp.buttons[15]?.pressed) this.setGear('N');

    this.lastGamepadButtons.clear();
    for (let i = 0; i < gp.buttons.length; i++) {
      if (gp.buttons[i]?.pressed) this.lastGamepadButtons.add(i);
    }

    return { steer, throttle, brake: handbrake ? 1 : brake, reset };
  }

  public cycleCamera(step: number): void {
    const names = [...this.cameraCards.keys()];
    if (names.length <= 1) return;
    const index = names.indexOf(this.selectedCamera ?? this.telemetry.active_view ?? '');
    const next = index < 0
      ? (step < 0 ? names.length - 1 : 0)
      : (index + step + names.length) % names.length;
    this.selectCamera(names[next]);
  }

  private selectCamera(name: string): void {
    if (!this.cameraCards.has(name)) return;
    this.selectedCamera = name;
    this.renderCameraSelection();

    this.sendEvent({
      type: 'select_camera',
      view_index: [...this.cameraCards.keys()].indexOf(name),
      camera: name,
    });
  }

  private addCamera(name: string): void {
    const label = cameraLabels.get(name) ?? name;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'camera-card';
    button.dataset.camera = name;
    button.setAttribute('aria-label', `切换到${label}`);
    button.addEventListener('click', () => this.selectCamera(name));

    const thumb = document.createElement('div');
    thumb.className = 'camera-thumb';
    const image = document.createElement('img');
    image.alt = `${label}模型预览`;
    image.hidden = true;
    const waiting = document.createElement('span');
    waiting.textContent = '等待模型画面';
    thumb.append(image, waiting);

    const badge = document.createElement('span');
    badge.className = 'camera-badge';
    badge.textContent = label;
    button.append(thumb, badge);
    document.getElementById('camera-list')?.append(button);
    this.cameraCards.set(name, { button, image, waiting });
  }

  private setCameraList(names: string[]): void {
    const existing = [...this.cameraCards.keys()];
    if (names.length === existing.length && names.every((name, index) => name === existing[index])) return;
    document.getElementById('camera-list')?.replaceChildren();
    this.cameraCards.clear();
    for (const name of names) this.addCamera(name);
    if (this.selectedCamera && !this.cameraCards.has(this.selectedCamera)) this.selectedCamera = undefined;
  }

  private renderCameraSelection(): void {
    const count = this.cameraCards.size;
    const active = this.selectedCamera ?? this.telemetry.active_view;
    const selected = count === 1 ? this.cameraCards.keys().next().value : active;
    const countLabel = document.getElementById('camera-count');
    if (countLabel) countLabel.textContent = count ? `${count} 个视角` : '等待相机';
    const tag = document.getElementById('view-tag-label');
    if (tag) {
      tag.textContent = selected && this.cameraCards.has(selected)
        ? cameraLabels.get(selected) ?? selected
        : active === 'all' && count ? '全部视角' : '相机画面';
    }
    for (const [name, { button }] of this.cameraCards) {
      button.classList.toggle('active', name === selected);
      button.setAttribute('aria-pressed', String(name === selected));
    }
  }

  public shiftStep(step: number): void {
    const gears: Gear[] = ['P', 'R', 'N', 'D'];
    const currentIdx = gears.indexOf(this.currentGear);
    const nextIdx = Math.max(0, Math.min(gears.length - 1, currentIdx + step));
    this.setGear(gears[nextIdx]);
  }

  private startControlLoop(): void {
    this.loopInterval = window.setInterval(() => {
      if (!document.hidden && document.hasFocus()) this.updateControls();
    }, 33);
  }

  private updateControls(): void {
    const gp = this.pollGamepad();
    this.currentSteer = (this.pressedKeys.has('d') ? 1 : 0) - (this.pressedKeys.has('a') ? 1 : 0);
    if (gp && Math.abs(gp.steer) > 0.02) this.currentSteer = gp.steer;
    this.currentThrottle = Math.max(this.pressedKeys.has('w') ? 1 : 0, gp?.throttle || 0);
    this.currentBrake = Math.max(this.pressedKeys.has('s') || this.pressedKeys.has(' ') ? 1 : 0, gp?.brake || 0);

    // High-fidelity automotive haptic evaluation (ABS, launch torque, gear shift)
    const handbrake = this.pressedKeys.has(' ') || Boolean(gp && gp.brake > 0.9);
    this.haptics.update(this.currentBrake, this.currentThrottle, this.currentSpeedKmh, handbrake);

    if (gp?.reset && !this.lastGamepadReset) this.resetSimulation();
    this.lastGamepadReset = Boolean(gp?.reset);
    this.sendDriverCommand();
    this.renderTelemetry();
  }

  private sendDriverCommand(): void {
    const command = {
      type: 'game_wheel',
      action: 'state',
      // Browser/Gamepad X is right-positive; the backend converts to left-positive.
      steering: this.currentSteer,
      throttle: this.currentGear === 'P' || this.currentGear === 'N' ? 0 : this.currentThrottle,
      brake: this.currentGear === 'P' ? 1 : this.currentBrake,
      reverse: this.currentGear === 'R',
      gear: this.currentGear,
      buttons: [],
    };
    const encoded = JSON.stringify(command);
    if (encoded !== this.lastCommand && this.sendEvent(command)) this.lastCommand = encoded;
  }

  private connectionStatus(message: string, failed = false): void {
    const status = document.getElementById('connection-status');
    if (status) {
      status.textContent = message;
      status.dataset.state = failed ? 'error' : 'info';
    }
  }

  private async initWebRTC(): Promise<void> {
    this.releaseControls();
    this.peer?.close();
    this.videoEl.srcObject = null;
    this.selectedCamera = undefined;
    this.setCameraList([]);
    this.telemetry = {};
    this.renderCameraSelection();
    this.hasTelemetry = false;
    this.warningActive = false;
    this.lastCommand = '';
    this.renderTelemetry();
    this.connectionStatus('正在连接 FlashDreams…');
    const tunnel = new URLSearchParams(window.location.hash.slice(1)).get('turn');
    const peer = new RTCPeerConnection(tunnel ? {
      iceTransportPolicy: 'relay',
      iceServers: [{ urls: 'turn:127.0.0.1:13478?transport=tcp', username: 'flashdreams', credential: tunnel }],
    } : undefined);
    this.peer = peer;
    const channel = peer.createDataChannel('controls');
    this.controlChannel = channel;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      channel.onopen = () => {
        if (this.peer !== peer) return;
        this.connectionStatus('控制通道已连接，等待模型画面…');
        this.lastCommand = '';
        if (document.hidden || !document.hasFocus()) this.releaseControls();
        else this.updateControls();
      };
      channel.onclose = () => {
        if (this.peer === peer) {
          this.hasTelemetry = false;
          this.telemetry = {};
          this.setCameraList([]);
          this.renderCameraSelection();
          this.renderTelemetry();
          this.connectionStatus('控制连接已断开，请在设置中重连', true);
        }
      };
      channel.onmessage = (event) => {
        if (this.peer !== peer) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'error') {
            this.connectionStatus(`后端拒绝消息：${data.message}`, true);
          } else if (data.type === 'recording') {
            this.renderRecording(data);
          } else if (data.type === 'camera_preview') {
            if (typeof data.camera !== 'string' || !data.camera ||
                typeof data.image !== 'string' || !data.image.startsWith('data:image/jpeg;base64,')) return;
            if (!this.cameraCards.has(data.camera) && !Array.isArray(this.telemetry.camera_names)) {
              this.addCamera(data.camera);
              this.renderCameraSelection();
            }
            const card = this.cameraCards.get(data.camera);
            if (card) {
              card.image.src = data.image;
              card.image.hidden = false;
              card.waiting.hidden = true;
            }
          } else if (data.type === 'telemetry') {
            this.telemetry = data;
            if (Array.isArray(data.camera_names)) {
              this.setCameraList([...new Set<string>(data.camera_names.filter(
                (name: unknown) => typeof name === 'string' && name.length > 0,
              ))]);
            }
            this.renderCameraSelection();
            if (Number.isFinite(data.speed_kmh)) {
              this.currentSpeedKmh = data.speed_kmh;
              this.hasTelemetry = true;
            }
            // Applied controls are telemetry, never a new driver command.
            this.warningActive = Boolean(data.warning);
            const title = document.getElementById('alert-title-text');
            if (title) title.textContent = data.warning || '';
            this.renderTelemetry();
          }
        } catch {
          this.connectionStatus('收到无法解析的后端消息', true);
        }
      };
      peer.onconnectionstatechange = () => {
        if (this.peer !== peer) return;
        if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) {
          this.connectionStatus(`WebRTC ${peer.connectionState}，请在设置中重连`, true);
        }
      };
      peer.addTransceiver('video', { direction: 'recvonly' });
      peer.ontrack = (event) => {
        if (this.peer !== peer) return;
        this.videoEl.srcObject = event.streams[0] || new MediaStream([event.track]);
        this.videoEl.onloadeddata = () => {
          if (this.peer === peer) this.connectionStatus('视频与控制已连接');
        };
        this.videoEl.play().catch(() => this.connectionStatus('点击画面开始播放'));
      };
      await peer.setLocalDescription(await peer.createOffer());
      if (peer.iceGatheringState !== 'complete') {
        await new Promise<void>((resolve, reject) => {
          const done = () => {
            if (peer.iceGatheringState === 'complete') {
              clearTimeout(timeout);
              peer.removeEventListener('icegatheringstatechange', done);
              resolve();
            }
          };
          peer.addEventListener('icegatheringstatechange', done);
          timeout = setTimeout(() => {
            peer.removeEventListener('icegatheringstatechange', done);
            reject(new Error('ICE 地址收集超时'));
          }, 15000);
          done();
        });
      }
      if (this.peer !== peer) return;
      const response = await fetch(`${this.backendUrl}/api/webrtc/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(peer.localDescription),
        signal: AbortSignal.timeout(30000),
      });
      if (!response.ok) throw new Error(`后端 HTTP ${response.status}: ${(await response.text()).slice(0, 180)}`);
      const answer = await response.json();
      if (this.peer !== peer) return;
      await peer.setRemoteDescription(answer);
    } catch (error) {
      if (this.peer === peer) {
        channel.onclose = null;
        peer.onconnectionstatechange = null;
        this.controlChannel = null;
        peer.close();
        this.connectionStatus(`连接失败：${error instanceof Error ? error.message : String(error)}。确认后端 8089 已启动后重连。`, true);
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  private bindUI(): void {
    // Click gear items to shift
    const gearItems = document.querySelectorAll('.gear-item');
    gearItems.forEach((item) => {
      item.addEventListener('click', () => {
        const g = item.getAttribute('data-gear') as Gear;
        if (g) this.setGear(g);
      });
    });

    // Alert modal dismiss
    const hotlineBtn = document.getElementById('alert-action-btn');
    if (hotlineBtn) {
      hotlineBtn.addEventListener('click', () => {
        this.warningActive = false;
        const card = document.getElementById('alert-card');
        if (card) {
          card.classList.add('entering');
          setTimeout(() => {
            card.style.display = 'none';
          }, 240);
        }
      });
    }

    // Reset button on stage
    document.getElementById('reset-stage-btn')?.addEventListener('click', () => {
      this.resetSimulation();
    });

    // Fullscreen toggle
    document.getElementById('fullscreen-btn')?.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
  }

  private bindSettings(): void {
    const overlay = document.getElementById('settings-modal-overlay');
    const openBtn = document.getElementById('settings-open-btn');
    const closeBtn = document.getElementById('settings-close-btn');

    openBtn?.addEventListener('click', () => overlay?.classList.remove('hidden'));
    closeBtn?.addEventListener('click', () => overlay?.classList.add('hidden'));

    overlay?.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.classList.add('hidden');
    });

    // Server reconnect
    document.getElementById('btn-reconnect-server')?.addEventListener('click', () => {
      const input = document.getElementById('setting-backend-url') as HTMLInputElement;
      const value = input?.value.trim().replace(/\/$/, '') || '';
      if (value) {
        try {
          const url = new URL(value);
          if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
        } catch {
          this.connectionStatus('后端地址必须是 http:// 或 https:// 地址', true);
          return;
        }
      }
      this.backendUrl = value;
      void this.initWebRTC();
      overlay?.classList.add('hidden');
    });

    // Template selection & addition
    const templateCards = document.querySelectorAll('.template-card');
    templateCards.forEach((card) => {
      card.addEventListener('click', () => {
        templateCards.forEach((c) => c.classList.remove('active'));
        card.classList.add('active');
      });
    });

    document.getElementById('btn-add-template')?.addEventListener('click', () => {
      const name = window.prompt('请输入新座舱/仿真模板名称:', '自定义多相机工作台');
      if (name) {
        const grid = document.getElementById('template-slot-grid');
        const btn = document.getElementById('btn-add-template');
        if (grid && btn) {
          const newCard = document.createElement('div');
          newCard.className = 'template-card';
          newCard.innerHTML = `<span class="template-name">${name}</span><span class="template-desc">自定义导入模板</span>`;
          newCard.addEventListener('click', () => {
            document.querySelectorAll('.template-card').forEach((c) => c.classList.remove('active'));
            newCard.classList.add('active');
          });
          grid.insertBefore(newCard, btn);
        }
      }
    });

    const denoiseSelect = document.getElementById('select-denoise-steps') as HTMLSelectElement;
    if (denoiseSelect) {
      denoiseSelect.disabled = true;
      denoiseSelect.title = '请通过后端启动参数 --denoise-steps 设置';
    }

    const vibrateCheck = document.getElementById('check-gamepad-vibrate') as HTMLInputElement;
    if (vibrateCheck) {
      vibrateCheck.checked = true;
      vibrateCheck.addEventListener('change', () => {
        this.haptics.setEnabled(vibrateCheck.checked);
        if (vibrateCheck.checked) {
          this.haptics.gearShift();
        }
      });
    }
  }

  private renderTelemetry(): void {
    ['p', 'r', 'n', 'd'].forEach(g => {
      const item = document.getElementById(`gear-${g}`);
      item?.classList.toggle('active', this.currentGear.toLowerCase() === g);
      item?.setAttribute('aria-pressed', String(this.currentGear.toLowerCase() === g));
    });

    // Top-left OEM dash cluster (Image 6 style)
    const topSpeed = document.getElementById('top-speed-val');
    if (topSpeed) topSpeed.textContent = this.hasTelemetry ? Math.round(this.currentSpeedKmh).toString() : '0';

    const topGear = document.getElementById('top-gear-box');
    if (topGear) topGear.textContent = this.currentGear;
    // Top bar cluster: only active states are bold, others unlit dim gray
    const pIcon = document.getElementById('top-p-icon');
    if (pIcon) {
      pIcon.classList.toggle('active', this.currentGear === 'P');
    }

    const autohold = document.getElementById('warn-autohold');
    if (autohold) {
      autohold.classList.toggle('active', this.currentSpeedKmh < 0.5 && this.currentBrake > 0.3);
    }

    const masterWarn = document.getElementById('warn-master');
    if (masterWarn) {
      masterWarn.classList.toggle('active', this.warningActive);
    }

    const card = document.getElementById('alert-card');
    if (card) {
      card.style.display = this.warningActive ? 'flex' : 'none';
    }

    // 1. Speed
    const speedEl = document.getElementById('speed-num');
    if (speedEl) speedEl.textContent = this.hasTelemetry ? Math.round(this.currentSpeedKmh).toString() : '—';

    // 2. Steering Wheel rotation
    const steerDeg = this.currentSteer * 45;
    const wheel = document.getElementById('steering-wheel');
    if (wheel) {
      wheel.style.transform = `rotate(${steerDeg}deg)`;
    }
    const steerAngleText = document.getElementById('steer-angle-val');
    if (steerAngleText) {
      steerAngleText.textContent = `${this.currentSteer > 0 ? '+' : ''}${(this.currentSteer * 270).toFixed(1)}°`;
    }

    // 3. Throttle & Brake bars
    const throttle = this.currentGear === 'P' || this.currentGear === 'N' ? 0 : this.currentThrottle;
    const brake = this.currentGear === 'P' ? 1 : this.currentBrake;
    const throttleBar = document.getElementById('throttle-fill');
    if (throttleBar) throttleBar.style.width = `${Math.round(throttle * 100)}%`;
    const throttleVal = document.getElementById('throttle-val');
    if (throttleVal) throttleVal.textContent = `${Math.round(throttle * 100)}%`;

    const brakeBar = document.getElementById('brake-fill');
    if (brakeBar) brakeBar.style.width = `${Math.round(brake * 100)}%`;
    const brakeVal = document.getElementById('brake-val');
    if (brakeVal) brakeVal.textContent = `${Math.round(brake * 100)}%`;

    // Dual Power / Brake HUD Bar under speed
    const hudThrottle = document.getElementById('hud-throttle-fill');
    if (hudThrottle) hudThrottle.style.width = `${Math.round(throttle * 50)}%`;
    const hudBrake = document.getElementById('hud-brake-fill');
    if (hudBrake) hudBrake.style.width = `${Math.round(brake * 50)}%`;

    // OEM Pedal & Steering icons (Image 6 style)
    const oemBrake = document.getElementById('oem-brake-svg');
    if (oemBrake) oemBrake.classList.toggle('active-brake', brake > 0.04);
    const oemThrottle = document.getElementById('oem-throttle-svg');
    if (oemThrottle) oemThrottle.classList.toggle('active-throttle', throttle > 0.04);
    const oemSteer = document.getElementById('oem-steer-svg');
    if (oemSteer) oemSteer.style.transform = `rotate(${this.currentSteer * 45}deg)`;

    // Hide connection toast once connected
    const conn = document.getElementById('connection-status');
    if (conn && this.peer && this.peer.connectionState === 'connected' && conn.style.display !== 'none') {
      conn.style.display = 'none';
    }

    // 4. World Scenario Native Image Stream
    this.updateWorldScenarioView();
  }

  /**
   * World Scenario uses real image streamed directly from backend service
   */
  private updateWorldScenarioView(): void {
    const bevImg = document.getElementById('bev-image') as HTMLImageElement;
    const placeholder = document.getElementById('bev-placeholder');

    if (this.telemetry.bev_image) {
      if (bevImg) {
        if (bevImg.getAttribute('src') !== this.telemetry.bev_image) bevImg.src = this.telemetry.bev_image;
        bevImg.style.display = 'block';
      }
      if (placeholder) {
        placeholder.style.display = 'none';
      }
    } else {
      if (bevImg) {
        bevImg.removeAttribute('src');
        bevImg.style.display = 'none';
      }
      if (placeholder) placeholder.style.display = 'flex';
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new CockpitApp();
});
