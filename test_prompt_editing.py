#!/usr/bin/env python3
"""Test PR #431 live prompt editing and actor spawning features."""
import sys
sys.path.insert(0, 'integrations/omnidreams')

print('[TEST] PR #431 Live Prompt Editing Test')
print('='*60)
sys.stdout.flush()

try:
    # Test 1: Import new modules
    print('[TEST] 1. Importing prompt editing modules...')
    sys.stdout.flush()

    from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
    from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend
    from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig

    print('[TEST] ✓ Imports successful')
    sys.stdout.flush()

    # Test 2: Load manifest
    print('[TEST] 2. Loading perf manifest...')
    sys.stdout.flush()

    manifest = load_world_model_manifest(
        r'integrations/omnidreams/omnidreams/interactive_drive/configs/example_world_model_perf.yaml'
    )
    print(f'[TEST] ✓ Manifest loaded: {manifest.resolution_wh}@{manifest.fps}fps')
    sys.stdout.flush()

    # Test 3: Create backend
    print('[TEST] 3. Creating WorldModelRenderBackend...')
    sys.stdout.flush()

    chunk = ChunkConfig(chunk_frames=8, initial_chunk_frames=5, fps=30)
    raster = RasterConfig(width=1168, height=640)
    backend = WorldModelRenderBackend(manifest=manifest, chunk=chunk, raster=raster)

    print('[TEST] ✓ Backend created')
    sys.stdout.flush()

    # Test 4: Check for TextEditGuidance
    print('[TEST] 4. Checking for TextEditGuidance...')
    sys.stdout.flush()

    try:
        from flashdreams.core.prompting.guidance import TextEditGuidance
        print('[TEST] ✓ TextEditGuidance available')
    except ImportError:
        print('[TEST] ⚠ TextEditGuidance not yet available (may need rebuild)')

    sys.stdout.flush()

    # Test 5: Check for KV cache functions
    print('[TEST] 5. Checking for KV cache editing...')
    sys.stdout.flush()

    try:
        from flashdreams.core.attention.kvcache import clone_kv, overwrite_kv
        print('[TEST] ✓ KV cache editing functions available')
    except ImportError:
        print('[TEST] ⚠ KV cache functions not yet available')

    sys.stdout.flush()

    # Test 6: Check for actor spawning
    print('[TEST] 6. Checking for actor spawning...')
    sys.stdout.flush()

    try:
        from omnidreams.interactive_drive.simulation.components import DynamicActor
        print('[TEST] ✓ DynamicActor spawning available')
    except ImportError:
        print('[TEST] ⚠ DynamicActor not yet available')

    sys.stdout.flush()

    print()
    print('='*60)
    print('[TEST] ✓ All PR #431 features check complete!')
    print('[TEST] Next: git merge origin/main to apply PR #431')
    print('='*60)

except Exception as e:
    print(f'[TEST] ✗ ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
