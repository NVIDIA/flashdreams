#!/usr/bin/env python3
"""Minimal test to isolate warmup error."""
import sys
import traceback
sys.path.insert(0, 'integrations/omnidreams')

print("[TEST] Starting minimal warmup test", flush=True)

try:
    from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
    from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend
    from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig

    print("[TEST] Imports done", flush=True)

    manifest = load_world_model_manifest(
        r'integrations\omnidreams\omnidreams\interactive_drive\configs\example_world_model_perf.yaml'
    )
    print("[TEST] Manifest loaded", flush=True)

    chunk = ChunkConfig(chunk_frames=8, initial_chunk_frames=5, fps=30)
    raster = RasterConfig(width=1168, height=640)
    backend = WorldModelRenderBackend(manifest=manifest, chunk=chunk, raster=raster)
    print("[TEST] Backend created", flush=True)

    print("[TEST] >>> CALLING warmup_model() <<<", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    backend.warmup_model()

    print("[TEST] ✓ warmup_model() completed successfully", flush=True)

except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
    print("[TRACEBACK]", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
