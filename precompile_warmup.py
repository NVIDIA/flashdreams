#!/usr/bin/env python3
"""Warmup torch.compile cache for interactive-drive perf."""
print("[START] Script started, before any imports", flush=True)
import sys
print("[START] sys imported", flush=True)
sys.stdout.flush()
import time
print("[START] time imported", flush=True)
sys.stdout.flush()
sys.path.insert(0, 'integrations/omnidreams')
print("[START] sys.path modified", flush=True)
sys.stdout.flush()

def log(msg):
    elapsed = time.time() - start
    print(f'[{elapsed:7.2f}s] {msg}', flush=True)

start = time.time()
log('[PRECOMPILE] Loading manifest...')
from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
log('[PRECOMPILE] Manifest imported')

log('[PRECOMPILE] Loading YAML config...')
import sys as _sys
print("[YAML-LOAD] About to call load_world_model_manifest", flush=True)
_sys.stdout.flush()
_sys.stderr.flush()
manifest = load_world_model_manifest(
    r'integrations/omnidreams/omnidreams/interactive_drive/configs/example_world_model_perf.yaml'
)
print("[YAML-LOAD] load_world_model_manifest returned", flush=True)
_sys.stdout.flush()
_sys.stderr.flush()
log(f'[PRECOMPILE] YAML loaded (res={manifest.resolution_wh}, fps={manifest.fps})')

log('[PRECOMPILE] Importing backend classes...')
print('[IMPORT] >>> ABOUT TO IMPORT WorldModelRenderBackend <<<', flush=True)
sys.stdout.flush()
from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend
print('[IMPORT] >>> WorldModelRenderBackend IMPORTED <<<', flush=True)
sys.stdout.flush()
print('[IMPORT] >>> ABOUT TO IMPORT ChunkConfig, RasterConfig <<<', flush=True)
sys.stdout.flush()
from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig
print('[IMPORT] >>> ChunkConfig, RasterConfig IMPORTED <<<', flush=True)
sys.stdout.flush()
log('[PRECOMPILE] Backend classes imported')

log('[PRECOMPILE] Creating chunk config...')
chunk = ChunkConfig(chunk_frames=8, initial_chunk_frames=5, fps=30)
log('[PRECOMPILE] Chunk config created')

log('[PRECOMPILE] Creating raster config...')
raster = RasterConfig(width=1168, height=640)
log('[PRECOMPILE] Raster config created')

log('[PRECOMPILE] Creating WorldModelRenderBackend (loading models)...')
print('>>> ABOUT TO CREATE BACKEND <<<', flush=True)
sys.stdout.flush()
import sys as sys2
sys2.stderr.flush()
try:
    print(f'[{time.time()-start:.2f}s] Creating backend instance...', flush=True)
    backend = WorldModelRenderBackend(manifest=manifest, chunk=chunk, raster=raster)
    print(f'[{time.time()-start:.2f}s] >>> BACKEND CREATED SUCCESSFULLY <<<', flush=True)
    log('[PRECOMPILE] Backend created - models loaded')
except Exception as e:
    print(f'[{time.time()-start:.2f}s] ERROR: {type(e).__name__}: {e}', flush=True)
    log(f'[PRECOMPILE] ERROR during backend creation: {type(e).__name__}')
    raise

import platform as _platform
if _platform.system() == "Windows":
    log('[PRECOMPILE] === SKIPPING WARMUP ON WINDOWS (torch.compile hangs) ===')
    log('[PRECOMPILE] Models cached. App will run without torch.compile on Windows.')
else:
    log('[PRECOMPILE] === STARTING TORCH.COMPILE WARMUP ===')
    log('[PRECOMPILE] Calling backend.warmup_model()...')
    try:
        backend.warmup_model()
        log('[PRECOMPILE] ✓ Warmup complete')
    except Exception as e:
        import traceback
        log(f'[PRECOMPILE] ERROR in warmup: {type(e).__name__}: {e}')
        traceback.print_exc()
        raise

log('[PRECOMPILE] === COMPILATION CACHED TO DISK ===')
log('[PRECOMPILE] ✓ SETUP COMPLETE - torch.compile cached')
log(f'[PRECOMPILE] Total time: {time.time()-start:.2f}s')
