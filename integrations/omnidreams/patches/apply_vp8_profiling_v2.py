#!/usr/bin/env python3
"""Apply VP8 encode profiling using the flashdreams profiler directly.

Uses the same profiler as the rest of the pipeline — events appear in
the same JSONL file with aligned timestamps.

Usage (inside Docker with venv activated):
    python3 apply_vp8_profiling_v2.py

To revert:
    cp $AIORTC_DIR/rtcrtpsender.py.bak $AIORTC_DIR/rtcrtpsender.py
"""

import importlib
import os
import shutil
import sys


PROFILING_BLOCK = r"""
# ---------- VP8 encode profiling (temporary) ----------
try:
    from flashdreams.serving.webrtc import profiler as _wp
except ImportError:
    _wp = None

def _vp8_encode_with_profiling(encoder, data, force_keyframe):
    if _wp is not None and _wp.is_enabled():
        with _wp.measure("vp8_encode"):
            return encoder.encode(data, force_keyframe)
    return encoder.encode(data, force_keyframe)
# ---------- end VP8 encode profiling ----------
"""

OLD_ENCODE_CALL = (
    "            payloads, timestamp = await self.__loop.run_in_executor(\n"
    "                None, self.__encoder.encode, data, force_keyframe\n"
    "            )"
)

NEW_ENCODE_CALL = (
    "            payloads, timestamp = await self.__loop.run_in_executor(\n"
    "                None, _vp8_encode_with_profiling, self.__encoder, data, force_keyframe\n"
    "            )"
)

ANCHOR_LINE = "logger = logging.getLogger(__name__)"


def main():
    spec = importlib.util.find_spec("aiortc")
    if spec is None or spec.submodule_search_locations is None:
        print("ERROR: Cannot find aiortc package. Is the venv activated?", file=sys.stderr)
        sys.exit(1)

    aiortc_dir = spec.submodule_search_locations[0]
    target = os.path.join(aiortc_dir, "rtcrtpsender.py")
    backup = target + ".bak"

    if not os.path.exists(target):
        print(f"ERROR: {target} not found", file=sys.stderr)
        sys.exit(1)

    with open(target, "r") as f:
        content = f.read()

    if "_vp8_encode_with_profiling" in content:
        print("Patch already applied. To re-apply, restore from .bak first:")
        print(f"  cp {backup} {target}")
        sys.exit(0)

    if OLD_ENCODE_CALL not in content:
        print("ERROR: Could not find the encode call to patch.", file=sys.stderr)
        sys.exit(1)

    if ANCHOR_LINE not in content:
        print(f"ERROR: Could not find anchor line.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"Backed up: {backup}")

    content = content.replace(ANCHOR_LINE, PROFILING_BLOCK + "\n" + ANCHOR_LINE, 1)
    content = content.replace(OLD_ENCODE_CALL, NEW_ENCODE_CALL, 1)

    with open(target, "w") as f:
        f.write(content)

    # Clear bytecode cache
    cache_dir = os.path.join(aiortc_dir, "__pycache__")
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("rtcrtpsender") and fname.endswith(".pyc"):
                os.remove(os.path.join(cache_dir, fname))
                print(f"Removed cached: {fname}")

    print(f"Patched: {target}")
    print("Verify: python3 -c \"from aiortc.rtcrtpsender import _vp8_encode_with_profiling; print('OK')\"")


if __name__ == "__main__":
    main()
