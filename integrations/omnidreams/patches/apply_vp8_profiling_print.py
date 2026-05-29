#!/usr/bin/env python3
"""Apply VP8 encode print-based profiling to aiortc's rtcrtpsender.py.

Prints per-frame VP8 encode timing to stdout every 100 frames.
No file I/O — avoids conflicts with the flashdreams profiler.

Usage (inside Docker with venv activated):
    python3 apply_vp8_profiling_print.py

To revert:
    cp $AIORTC_DIR/rtcrtpsender.py.bak $AIORTC_DIR/rtcrtpsender.py
"""

import importlib
import os
import shutil
import sys


PROFILING_BLOCK = r"""
# ---------- VP8 encode profiling (temporary, print-based) ----------
_vp8_durs = []

def _vp8_encode_with_profiling(encoder, data, force_keyframe):
    t0 = time.perf_counter()
    result = encoder.encode(data, force_keyframe)
    dur_ms = (time.perf_counter() - t0) * 1000.0
    _vp8_durs.append(dur_ms)
    n = len(_vp8_durs)
    if n % 100 == 0:
        durs = sorted(_vp8_durs)
        avg = sum(durs) / n
        p50 = durs[n // 2]
        p95 = durs[int(n * 0.95)]
        print(
            f"[VP8 ENCODE] n={n} avg={avg:.2f}ms p50={p50:.2f}ms "
            f"p95={p95:.2f}ms min={durs[0]:.2f}ms max={durs[-1]:.2f}ms",
            flush=True,
        )
    return result
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
    print("Run the server and look for [VP8 ENCODE] lines in stdout.")


if __name__ == "__main__":
    main()
