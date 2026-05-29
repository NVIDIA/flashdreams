#!/usr/bin/env python3
"""Apply VP8 encode profiling patch to aiortc's rtcrtpsender.py.

Usage (inside Docker with venv activated):
    python3 /path/to/apply_vp8_profiling.py

To revert:
    cp rtcrtpsender.py.bak rtcrtpsender.py
"""

import importlib
import os
import shutil
import sys

PROFILING_BLOCK = '''
# ---------- VP8 encode profiling (temporary) ----------
import json as _json
import os as _os
import threading as _threading
from pathlib import Path as _Path

_VP8_PROF_ENABLED = _os.environ.get("WEBRTC_PROFILE", "").strip() in ("1", "true", "yes")
_VP8_PROF_PATH = _Path(
    _os.environ.get("WEBRTC_PROFILE_PATH", "/tmp/webrtc_profile.jsonl")
)
_vp8_prof_lock = _threading.Lock()
_vp8_prof_epoch: float = 0.0
_vp8_prof_file = None


def _vp8_prof_ensure_open():
    global _vp8_prof_file, _vp8_prof_epoch
    if _vp8_prof_file is not None:
        return
    _vp8_prof_epoch = time.perf_counter()
    _VP8_PROF_PATH.parent.mkdir(parents=True, exist_ok=True)
    _vp8_prof_file = open(_VP8_PROF_PATH, "a")


def _vp8_encode_with_profiling(encoder, data, force_keyframe):
    global _vp8_prof_file
    if not _VP8_PROF_ENABLED:
        return encoder.encode(data, force_keyframe)
    _vp8_prof_ensure_open()
    t0 = time.perf_counter()
    result = encoder.encode(data, force_keyframe)
    t1 = time.perf_counter()
    with _vp8_prof_lock:
        if _vp8_prof_file is not None:
            record = _json.dumps({
                "stage": "vp8_encode",
                "start": round(t0 - _vp8_prof_epoch, 6),
                "end": round(t1 - _vp8_prof_epoch, 6),
                "dur_ms": round((t1 - t0) * 1000.0, 3),
                "chunk": -1,
                "tid": _threading.current_thread().name,
            }, separators=(",", ":"))
            _vp8_prof_file.write(record + "\\n")
            _vp8_prof_file.flush()
    return result
# ---------- end VP8 encode profiling ----------
'''

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
        print("Patch already applied. To re-apply, restore from .bak first.")
        sys.exit(0)

    if OLD_ENCODE_CALL not in content:
        print("ERROR: Could not find the encode call to patch. File may have changed.", file=sys.stderr)
        print("Looking for:\n" + OLD_ENCODE_CALL, file=sys.stderr)
        sys.exit(1)

    if ANCHOR_LINE not in content:
        print(f"ERROR: Could not find anchor line '{ANCHOR_LINE}'", file=sys.stderr)
        sys.exit(1)

    # Back up
    shutil.copy2(target, backup)
    print(f"Backed up: {backup}")

    # Insert profiling block before the logger line
    content = content.replace(ANCHOR_LINE, PROFILING_BLOCK + "\n" + ANCHOR_LINE, 1)

    # Replace the encode call
    content = content.replace(OLD_ENCODE_CALL, NEW_ENCODE_CALL, 1)

    with open(target, "w") as f:
        f.write(content)

    print(f"Patched: {target}")
    print("Verify with: python3 -c \"from aiortc.rtcrtpsender import _vp8_encode_with_profiling; print('OK')\"")


if __name__ == "__main__":
    main()
