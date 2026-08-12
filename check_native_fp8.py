#!/usr/bin/env python3
"""Check if native FP8 acceleration is available and enabled."""
import sys
sys.path.insert(0, 'integrations/omnidreams')

print("[CHECK] Testing native FP8 availability...")
sys.stdout.flush()

try:
    from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
    manifest_path = r"C:\workspace\world\flashdream_public\integrations\omnidreams\omnidreams\interactive_drive\configs\example_world_model_perf.yaml"
    manifest = load_world_model_manifest(manifest_path)

    print(f"[CHECK] native_dit_acceleration: {manifest.native_dit_acceleration}")
    print(f"[CHECK] native_dit_backend: {manifest.native_dit_backend}")
    print(f"[CHECK] native_dit_attention_backend: {manifest.native_dit_attention_backend}")
    sys.stdout.flush()

    # Try to import the native module
    print("[CHECK] Attempting to import native acceleration module...")
    sys.stdout.flush()

    try:
        from omnidreams.native.acceleration import NativeAccelerationConfig, require_extension_symbols
        from omnidreams.native import omnidreams_singleview
        print("[CHECK] ✓ Native module imported successfully")
        sys.stdout.flush()

        # Try to select backend
        print("[CHECK] Attempting to select optimized DiT backend...")
        sys.stdout.flush()
        native_config = NativeAccelerationConfig(mode=manifest.native_dit_acceleration)
        selection = omnidreams_singleview.select_backend('optimized_dit', native_config)

        if selection.enabled:
            print(f"[CHECK] ✓ Native FP8 ENABLED (backend={selection.backend})")
        else:
            print(f"[CHECK] ✗ Native FP8 DISABLED (backend={selection.backend})")
        sys.stdout.flush()

    except ImportError as e:
        print(f"[CHECK] ✗ Native module NOT available: {e}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CHECK] ✗ Backend selection failed: {type(e).__name__}: {e}")
        sys.stdout.flush()

except Exception as e:
    print(f"[CHECK] ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
