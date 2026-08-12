#!/usr/bin/env python3
"""Test warmup_model in isolation with detailed debug."""
import sys
import time
sys.path.insert(0, 'integrations/omnidreams')

print('[TEST] Starting isolated warmup test', flush=True)
start = time.time()

try:
    print(f'[TEST] [{time.time()-start:.2f}s] Importing manifest...', flush=True)
    from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
    manifest = load_world_model_manifest(
        r'integrations/omnidreams/omnidreams/interactive_drive/configs/example_world_model_perf.yaml'
    )
    print(f'[TEST] [{time.time()-start:.2f}s] Manifest loaded', flush=True)

    print(f'[TEST] [{time.time()-start:.2f}s] Importing FlashdreamsWorldModelSession...', flush=True)
    from omnidreams.interactive_drive.world_model.flashdreams_adapter import FlashdreamsWorldModelSession
    print(f'[TEST] [{time.time()-start:.2f}s] Session class imported', flush=True)

    print(f'[TEST] [{time.time()-start:.2f}s] Creating session...', flush=True)
    session = FlashdreamsWorldModelSession(manifest)
    print(f'[TEST] [{time.time()-start:.2f}s] Session created', flush=True)

    print(f'[TEST] [{time.time()-start:.2f}s] Calling warmup_model()...', flush=True)
    session.warmup_model()
    print(f'[TEST] [{time.time()-start:.2f}s] ✓ warmup_model() COMPLETE', flush=True)

except KeyboardInterrupt:
    print(f'[TEST] [{time.time()-start:.2f}s] INTERRUPTED by user', flush=True)
except Exception as e:
    print(f'[TEST] [{time.time()-start:.2f}s] ERROR: {type(e).__name__}: {e}', flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
