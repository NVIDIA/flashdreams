// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createServer } = require('node:http');
const { mkdtemp, mkdir, writeFile, symlink, rm } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join } = require('node:path');

test('local export API requires its session token, rejects cross-site requests and serves bounded ranges', async () => {
  const { createExportHandler } = await import('../export-api.mjs');
  const directory = await mkdtemp(join(tmpdir(), 'drive-export-api-'));
  const root = join(directory, 'exports');
  await mkdir(join(root, 'drive-test'), { recursive: true });
  await writeFile(join(root, 'drive-test.zip'), '0123456789');
  await writeFile(join(root, 'drive-test', 'manifest.json'), JSON.stringify({ frame_count: 2, fps: 10, status: 'ready' }));
  await writeFile(join(directory, 'outside.zip'), 'private');
  await symlink(join(directory, 'outside.zip'), join(root, 'drive-outside.zip'));
  const { handler, token } = createExportHandler({ root });
  const server = createServer((req, res) => { void handler(req, res, () => { res.writeHead(404); res.end(); }); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const headers = { Authorization: `Bearer ${token}` };
  try {
    assert.equal((await fetch(`${url}/api/drive-exports`)).status, 401);
    assert.equal((await fetch(`${url}/api/drive-exports`, { headers: { ...headers, Origin: 'https://untrusted.example' } })).status, 403);
    const listing = await (await fetch(`${url}/api/drive-exports`, { headers })).json();
    assert.equal(listing.recordings.length, 1);
    const response = await fetch(`${url}/api/drive-exports/drive-test.zip`, { headers: { ...headers, Range: 'bytes=2-5' } });
    assert.equal(response.status, 206); assert.equal(await response.text(), '2345');
    assert.equal(response.headers.get('content-range'), 'bytes 2-5/10');
    assert.equal((await fetch(`${url}/api/drive-exports/drive-outside.zip`, { headers })).status, 400);
    assert.equal((await fetch(`${url}/api/drive-exports/drive-test.zip`, { headers: { ...headers, Range: 'bytes=100-200' } })).status, 416);
    const invalid = await fetch(`${url}/api/comparison/render`, { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify({ right: '../outside' }) });
    assert.equal(invalid.status, 400);
  } finally { await new Promise(resolve => server.close(resolve)); await rm(directory, { recursive: true, force: true }); }
});
