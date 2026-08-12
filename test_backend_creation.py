#!/usr/bin/env python3
"""Test WorldModelRenderBackend creation in isolation."""
import sys
import time
sys.path.insert(0, 'integrations/omnidreams')

start = time.time()

def log(msg):
    print(f'[{time.time()-start:7.2f}s] {msg}', flush=True)

log('[TEST] Loading manifest...')
from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
manifest = load_world_model_manifest(
    r'integrations/omnidreams/omnidreams/interactive_drive/configs/example_world_model_perf.yaml'
)
log('[TEST] Manifest loaded')

log('[TEST] Importing backend...')
from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend
from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig
log('[TEST] Backend imported')

log('[TEST] Creating configs...')
chunk = ChunkConfig(chunk_frames=8, initial_chunk_frames=5, fps=30)
raster = RasterConfig(width=1168, height=640)
log('[TEST] Configs created')

log('[TEST] >>> CREATING BACKEND NOW <<<')
sys.stdout.flush()
try:
    backend = WorldModelRenderBackend(manifest=manifest, chunk=chunk, raster=raster)
    log('[TEST] >>> BACKEND CREATED SUCCESSFULLY <<<')
except Exception as e:
    log(f'[TEST] ERROR: {type(e).__name__}: {str(e)[:500]}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

log('[TEST] ✓ Backend creation test complete')
