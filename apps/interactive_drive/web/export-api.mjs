// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { createReadStream } from 'node:fs';
import { access, mkdir, readFile, readdir, realpath, stat, writeFile } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { randomBytes, timingSafeEqual } from 'node:crypto';

const safeId = /^[a-z0-9][a-z0-9-]{0,90}$/;
const exists = path => access(path).then(() => true, () => false);
const loopback = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

export function createExportHandler(options = {}) {
  const repo = options.repo || resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
  const root = resolve(options.root || process.env.DRIVE_EXPORT_DIR || resolve(repo, 'artifacts/drive-exports'));
  const gt = resolve(process.env.DRIVE_GT_PATH || resolve(repo, 'artifacts/jlu-123246900/jlu-20260714-123246900-raw-sixview'));
  const scene = resolve(process.env.DRIVE_SCENE_PATH || resolve(repo, 'artifacts/jlu-123246900/scene-rgb-campus'));
  const media = resolve(root, 'comparisons');
  const token = randomBytes(32).toString('hex');
  const jobs = new Map();
  let active = null;

  function json(res, status, data) {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(data));
  }

  async function serve(req, res, directory, name, type, attachment = false) {
    const actual = await realpath(resolve(directory, name));
    if (dirname(actual) !== await realpath(directory)) throw new Error('Invalid asset path');
    const info = await stat(actual);
    if (!info.isFile()) throw new Error('Not a file');
    let start = 0, end = info.size - 1, status = 200;
    const headers = { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'X-Content-Type-Options': 'nosniff' };
    if (attachment) headers['Content-Disposition'] = `attachment; filename="${basename(name)}"`;
    if (req.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!match || !match[1] && !match[2]) { res.writeHead(416); res.end(); return; }
      start = match[1] ? Number(match[1]) : Math.max(0, info.size - Number(match[2]));
      end = match[1] && match[2] ? Math.min(Number(match[2]), end) : end;
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= info.size) {
        res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }); res.end(); return;
      }
      status = 206; headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
    }
    headers['Content-Length'] = String(end - start + 1);
    res.writeHead(status, headers);
    if (req.method === 'HEAD') { res.end(); return; }
    const stream = createReadStream(actual, { start, end });
    stream.on('error', error => res.destroy(error));
    res.on('close', () => stream.destroy());
    stream.pipe(res);
  }

  const handler = async (req, res, next) => {
    const url = new URL(req.url || '/', 'http://localhost');
    if (!url.pathname.startsWith('/api/drive-exports') && !url.pathname.startsWith('/api/comparison')) { next(); return; }
    let reserved = false;
    try {
      const host = new URL(`http://${req.headers.host}`).hostname;
      if (!loopback.has(req.socket.remoteAddress) || !['localhost', '127.0.0.1', '[::1]'].includes(host)) {
        json(res, 403, { error: '导出仅允许本机访问，请通过 SSH 转发 Web 端口' }); return;
      }
      if (req.headers.origin && new URL(req.headers.origin).host !== req.headers.host) {
        json(res, 403, { error: '跨站请求已拒绝' }); return;
      }
      const supplied = req.headers.authorization?.replace(/^Bearer /, '') || url.searchParams.get('token') || '';
      if (Buffer.byteLength(supplied) !== Buffer.byteLength(token) || !timingSafeEqual(Buffer.from(supplied), Buffer.from(token))) {
        json(res, 401, { error: '会话凭证失效，请刷新页面' }); return;
      }
      if (url.pathname === '/api/drive-exports' && req.method === 'GET') {
        const names = await readdir(root).catch(error => { if (error.code === 'ENOENT') return []; throw error; });
        const recordings = [];
        for (const name of names.filter(name => name.endsWith('.zip') && safeId.test(name.slice(0, -4))).sort().reverse().slice(0, 100)) {
          const id = name.slice(0, -4), path = resolve(root, name);
          if (dirname(await realpath(path)) !== await realpath(root)) continue;
          const manifestPath = await realpath(resolve(root, id, 'manifest.json'));
          if (dirname(dirname(manifestPath)) !== await realpath(root)) continue;
          const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
          recordings.push({ id, frames: manifest.frame_count, fps: manifest.fps, state: manifest.status, bytes: (await stat(path)).size, url: `/api/drive-exports/${id}.zip` });
        }
        const files = await readdir(media).catch(error => { if (error.code === 'ENOENT') return []; throw error; });
        const videoFiles = await Promise.all(files.filter(name => name.endsWith('.mp4') && safeId.test(name.slice(0, -4))).map(async name => ({ name, time: (await stat(resolve(media, name))).mtimeMs })));
        const videos = videoFiles.sort((a, b) => b.time - a.time).slice(0, 12).map(({ name }) => ({ id: name.slice(0, -4), url: `/api/comparison/media/${name}` }));
        json(res, 200, { recordings, videos, referenceAvailable: await exists(resolve(gt, 'sample.json')) && await exists(resolve(scene, 'map.json')) }); return;
      }
      const download = /^\/api\/drive-exports\/([a-z0-9-]+)\.zip$/.exec(url.pathname);
      if (download && safeId.test(download[1]) && ['GET', 'HEAD'].includes(req.method)) {
        await serve(req, res, root, `${download[1]}.zip`, 'application/zip', true); return;
      }
      const asset = /^\/api\/comparison\/media\/([a-z0-9-]+)\.(mp4|jpg|json)$/.exec(url.pathname);
      if (asset && safeId.test(asset[1]) && ['GET', 'HEAD'].includes(req.method)) {
        await serve(req, res, media, `${asset[1]}.${asset[2]}`, { mp4: 'video/mp4', jpg: 'image/jpeg', json: 'application/json' }[asset[2]]); return;
      }
      if (url.pathname === '/api/comparison/template' && req.method === 'GET') {
        json(res, 200, JSON.parse(await readFile(resolve(repo, 'apps/interactive_drive/web/comparison-template.json'), 'utf8'))); return;
      }
      const jobMatch = /^\/api\/comparison\/jobs\/([a-z0-9-]+)$/.exec(url.pathname);
      if (jobMatch && req.method === 'GET') {
        const job = jobs.get(jobMatch[1]); json(res, job ? 200 : 404, job || { error: '制作任务不存在，请重新生成' }); return;
      }
      if (url.pathname === '/api/comparison/render' && req.method === 'POST') {
        if (active) { json(res, 409, { error: '已有视频正在制作，请等待完成' }); return; }
        active = 'preparing'; reserved = true;
        let body = '';
        for await (const chunk of req) { body += chunk.toString(); if (body.length > 8192) { json(res, 413, { error: '配置过大' }); return; } }
        const data = JSON.parse(body);
        if (data.right !== 'gt' && (typeof data.right !== 'string' || !safeId.test(data.right))) { json(res, 400, { error: '请选择有效的右侧素材' }); return; }
        const template = JSON.parse(await readFile(resolve(repo, 'apps/interactive_drive/web/comparison-template.json'), 'utf8'));
        for (const field of ['left_title', 'right_title', 'caption']) {
          if (data[field] !== undefined) {
            if (typeof data[field] !== 'string' || (field !== 'caption' && !data[field].trim()) || data[field].length > (field === 'caption' ? 130 : 58) || /[^\x20-\x7e]/.test(data[field])) { json(res, 400, { error: '视频文字请使用英文，标题不超过 58 字符，说明不超过 130 字符' }); return; }
            template[field] = data[field];
          }
        }
        if (data.right !== 'gt') {
          const folder = await realpath(resolve(root, data.right));
          if (dirname(folder) !== await realpath(root) || !await exists(resolve(root, `${data.right}.zip`))) { json(res, 400, { error: '请先停止录制，完成素材归档' }); return; }
        }
        const id = `comparison-${Date.now()}-${randomBytes(4).toString('hex')}`;
        await mkdir(media, { recursive: true });
        const templatePath = resolve(media, `${id}.template.json`);
        await writeFile(templatePath, JSON.stringify(template));
        const args = ['-m', 'interactive_drive.comparison', '--gt', gt, '--scene', scene, '--template', templatePath, '--output', resolve(media, `${id}.mp4`)];
        const egoModel = resolve(root, 'assets/haval-h6/ego-model.npz');
        if (await exists(egoModel)) args.push('--ego-model', egoModel);
        if (data.right !== 'gt') args.push('--right', resolve(root, data.right));
        const job = { id, state: 'running', frame: 0, frames: 0 };
        jobs.set(id, job); active = id;
        if (jobs.size > 64) jobs.delete(jobs.keys().next().value);
        const child = spawn(resolve(repo, '.venv/bin/python'), args, { cwd: repo, env: { ...process.env, CUDA_VISIBLE_DEVICES: '' }, stdio: ['ignore', 'pipe', 'pipe'] });
        let pending = '', errorText = '';
        child.stdout.on('data', chunk => {
          pending += chunk.toString(); const lines = pending.split('\n'); pending = lines.pop();
          for (const line of lines) { try { const progress = JSON.parse(line); if (Number.isInteger(progress.frame)) Object.assign(job, progress); } catch {} }
        });
        child.stderr.on('data', chunk => { errorText = (errorText + chunk.toString()).slice(-3000); });
        child.once('error', error => { job.state = 'error'; job.message = error.message; if (active === id) active = null; });
        child.once('close', code => {
          if (code === 0) { job.state = 'ready'; job.url = `/api/comparison/media/${id}.mp4`; job.poster = `/api/comparison/media/${id}.jpg`; }
          else { job.state = 'error'; job.message = errorText.trim().split('\n').at(-1) || job.message || '视频制作失败'; }
          if (active === id) active = null;
        });
        json(res, 202, job); return;
      }
      json(res, 404, { error: '接口不存在' });
    } catch (error) {
      if (!res.headersSent) json(res, error.code === 'ENOENT' ? 404 : 400, { error: error.message });
      else res.destroy(error);
    } finally {
      if (reserved && active === 'preparing') active = null;
    }
  };
  return { handler, token };
}

export default function driveExportPlugin() {
  const { handler, token } = createExportHandler();
  let build = false;
  const inject = html => html.replace('</head>', `<meta id="drive-export-token" name="drive-export-token" content="${token}"></head>`);
  return {
    name: 'drive-export',
    configResolved(config) { build = config.command === 'build'; },
    transformIndexHtml(html) { return build ? html : inject(html); },
    configureServer(server) { server.middlewares.use(handler); },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
      server.middlewares.use(async (req, res, next) => {
        const page = (req.url || '/').split('?')[0];
        if (!['/', '/index.html', '/comparison.html'].includes(page)) { next(); return; }
        try {
          const html = await readFile(resolve(server.config.root, server.config.build.outDir, page === '/comparison.html' ? 'comparison.html' : 'index.html'), 'utf8');
          res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(inject(html));
        } catch (error) { next(error); }
      });
    },
  };
}
