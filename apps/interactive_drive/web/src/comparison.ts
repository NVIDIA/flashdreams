// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export {};
const token = document.getElementById('drive-export-token')?.getAttribute('content') || '';
const signed = (path: string) => `${path}?token=${encodeURIComponent(token)}`;
const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const source = element<HTMLSelectElement>('right-source');
const video = element<HTMLVideoElement>('comparison-video');
const button = element<HTMLButtonElement>('render-button');
const status = element<HTMLParagraphElement>('render-status');
const progress = element<HTMLProgressElement>('render-progress');
let template: Record<string, string | number> = {};
let polling: ReturnType<typeof setTimeout> | undefined;

async function api(path: string, options?: RequestInit) {
  const response = await fetch(path, { ...options, headers: { ...options?.headers, Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 ${response.status}`);
  return data;
}

async function showVideo(path: string, poster?: string) {
  video.src = signed(path); video.hidden = false;
  if (poster) video.poster = signed(poster);
  element('preview-empty').hidden = true;
  const download = element<HTMLAnchorElement>('video-download');
  download.href = signed(path); download.hidden = false;
  element('preview-description').textContent = '视频已生成，可播放、拖动时间轴或下载 MP4。';
  const metadata = await api(path.replace(/\.mp4$/, '.json'));
  const gt = metadata.right === 'GT duplicate';
  element('mode-badge').textContent = gt ? 'GT × GT · 模板测试' : 'GT × SIM · 模拟对比';
  element('right-legend').textContent = gt ? '右侧：GT 副本' : '右侧：录制的模拟输出';
  const id = gt ? 'gt' : String(metadata.right).split('/').pop();
  if (Array.from(source.options).some(option => option.value === id)) source.value = id!;
  element<HTMLInputElement>('left-title').value = metadata.template.left_title;
  element<HTMLInputElement>('right-title').value = metadata.template.right_title;
  element<HTMLTextAreaElement>('video-caption').value = metadata.template.caption || '';
}

async function refresh() {
  const data = await api('/api/drive-exports');
  const selected = source.value;
  source.replaceChildren(new Option('GT 真实回放 · 测试模板', 'gt'));
  const list = element('recordings-list'); list.replaceChildren();
  element('recording-count').textContent = String(data.recordings.length);
  for (const recording of data.recordings) {
    if (recording.state === 'ready') source.add(new Option(`${recording.id} · ${recording.frames} 帧`, recording.id));
    const row = document.createElement('div'); row.className = 'recording-row';
    const name = document.createElement('strong'); name.textContent = recording.id;
    const detail = document.createElement('span'); detail.textContent = `${recording.frames} 帧 / ${recording.fps} FPS / ${(recording.bytes / 1024 / 1024).toFixed(1)} MB${recording.state === 'error' ? ' / 部分数据' : ''}`;
    const download = document.createElement('a'); download.className = 'text-link'; download.textContent = '下载素材 ZIP ↓'; download.href = signed(recording.url); download.download = `${recording.id}.zip`;
    row.append(name, detail, download); list.append(row);
  }
  if (Array.from(source.options).some(option => option.value === selected)) source.value = selected;
  if (!data.recordings.length) { const empty = document.createElement('p'); empty.className = 'empty-list'; empty.textContent = '在驾驶座舱点击「开始录制」，结束后点击「停止并归档」。'; list.append(empty); }
  element('reference-status').textContent = data.referenceAvailable ? '已找到六路真实 RGB 与场景地图。' : '未找到 GT 数据，请检查 Web 启动时的 DRIVE_GT_PATH 和 DRIVE_SCENE_PATH。';
  button.disabled = !data.referenceAvailable;
  if (!video.getAttribute('src') && data.videos.length) await showVideo(data.videos[0].url, data.videos[0].url.replace(/\.mp4$/, '.jpg'));
}

async function poll(id: string) {
  try {
    const job = await api(`/api/comparison/jobs/${id}`);
    progress.hidden = false;
    progress.value = job.frames ? job.frame / job.frames * 100 : 0;
    status.textContent = job.frames ? `正在制作 ${job.frame} / ${job.frames} 帧…` : '正在读取真实画面和场景地图…';
    if (job.state === 'running') { polling = setTimeout(() => { void poll(id); }, 1000); return; }
    sessionStorage.removeItem('comparison-job'); button.disabled = false;
    if (job.state === 'error') throw new Error(job.message);
    progress.value = 100; status.textContent = `已完成 ${job.frames} 帧，点击左侧播放或下载。`;
    await showVideo(job.url, job.poster);
  } catch (error) { status.textContent = error instanceof Error ? error.message : String(error); button.disabled = false; sessionStorage.removeItem('comparison-job'); }
}

source.addEventListener('change', () => {
  const gt = source.value === 'gt';
  element('mode-badge').textContent = gt ? 'GT × GT · 模板测试' : 'GT × SIM · 模拟对比';
  element('right-legend').textContent = gt ? '右侧：GT 副本' : '右侧：录制的模拟输出';
  element<HTMLInputElement>('right-title').value = gt ? String(template.right_title) : 'DriveWorld (Closed-loop Simulation)';
});
element('render-form').addEventListener('submit', event => {
  event.preventDefault(); button.disabled = true; progress.hidden = false; progress.value = 0;
  const settings = { right: source.value, left_title: element<HTMLInputElement>('left-title').value, right_title: element<HTMLInputElement>('right-title').value, caption: element<HTMLTextAreaElement>('video-caption').value };
  void api('/api/comparison/render', { method: 'POST', body: JSON.stringify(settings) }).then(job => {
    sessionStorage.setItem('comparison-job', job.id); void poll(job.id);
  }).catch(error => { status.textContent = error.message; button.disabled = false; });
});
element('refresh-recordings').addEventListener('click', () => { void refresh().catch(error => { status.textContent = error.message; }); });
element('template-download').addEventListener('click', () => {
  const settings = { ...template, left_title: element<HTMLInputElement>('left-title').value, right_title: element<HTMLInputElement>('right-title').value, caption: element<HTMLTextAreaElement>('video-caption').value };
  const url = URL.createObjectURL(new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'comparison-template.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('pagehide', () => clearTimeout(polling));
void api('/api/comparison/template').then(async data => {
  template = data;
  element<HTMLInputElement>('left-title').value = data.left_title;
  element<HTMLInputElement>('right-title').value = data.right_title;
  element<HTMLTextAreaElement>('video-caption').value = data.caption;
  await refresh();
  const pending = sessionStorage.getItem('comparison-job');
  if (pending) { button.disabled = true; void poll(pending); }
}).catch(error => { status.textContent = error.message; button.disabled = true; });
