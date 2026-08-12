#!/usr/bin/env python3
"""Standalone test for native DIT extension loading."""

import sys
import time
import os

os.chdir(r"C:\workspace\world\flashdream_public")
sys.path.insert(0, r"C:\workspace\world\flashdream_public")

from omnidreams.native.acceleration import NativeAccelerationConfig, NativeAccelerationMode
from omnidreams.native import omnidreams_singleview

print("[TEST] Starting native DIT extension load test...")
print()

try:
    print("[1/4] Loading optimized_dit Python module...")
    start = time.perf_counter()
    helper = omnidreams_singleview.load_python_module("optimized_dit")
    elapsed = time.perf_counter() - start
    print(f"✓ Loaded in {elapsed:.2f}s")
    print()

    print("[2/4] Creating NativeAccelerationConfig...")
    native_config = NativeAccelerationConfig(
        mode="required",  # string, not enum
        build_root=None,
        max_jobs=None,
        verbose_build=True,
    )
    print(f"✓ Config: mode={native_config.mode}")
    print()

    print("[3/4] Selecting backend (this will compile if needed)...")
    print("⏳ Starting compilation (may take 45-90 minutes on first run)...")
    print()
    start = time.perf_counter()
    selection = omnidreams_singleview.select_backend(
        "optimized_dit",
        native_config,
    )
    elapsed = time.perf_counter() - start
    print()
    print(f"✓ Backend selection completed in {elapsed:.2f}s")
    print(f"  Enabled: {selection.enabled}")
    print()

    if selection.enabled:
        print("[4/4] Loading extension (require_extension)...")
        start = time.perf_counter()
        ext = selection.require_extension()
        elapsed = time.perf_counter() - start
        print(f"✓ Extension loaded in {elapsed:.2f}s")
        print(f"  Extension: {ext}")
    else:
        print("[4/4] Backend disabled, skipping extension load")

    print()
    print("✓✓✓ SUCCESS - Native DIT extension ready ✓✓✓")

except Exception as e:
    print()
    print(f"✗✗✗ ERROR ✗✗✗")
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
