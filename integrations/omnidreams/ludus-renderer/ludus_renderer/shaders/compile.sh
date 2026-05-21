#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compile the EXT-style timestamped mesh-shader pipeline to SPIR-V and
# embed the bytecode into a C++ header for the Vulkan renderer.
#
# Prerequisites:
#   - glslangValidator (Vulkan SDK or distro package) on PATH, or
#     GLSLANG_VALIDATOR / VULKAN_SDK env var pointing at it.
#   - python3 with the standard library.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ -n "${GLSLANG_VALIDATOR:-}" ] && [ -x "$GLSLANG_VALIDATOR" ]; then
    GLSLANG="$GLSLANG_VALIDATOR"
elif [ -n "${VULKAN_SDK:-}" ] && [ -x "$VULKAN_SDK/bin/glslangValidator" ]; then
    GLSLANG="$VULKAN_SDK/bin/glslangValidator"
elif [ -n "${VULKAN_SDK:-}" ] && [ -x "$VULKAN_SDK/x86_64/bin/glslangValidator" ]; then
    GLSLANG="$VULKAN_SDK/x86_64/bin/glslangValidator"
else
    GLSLANG=$(command -v glslangValidator || true)
fi

if [ -z "${GLSLANG:-}" ]; then
    echo "error: glslangValidator not found. Install the Vulkan SDK or set GLSLANG_VALIDATOR." >&2
    exit 2
fi

echo "Using: $GLSLANG"
"$GLSLANG" --version 2>/dev/null | head -1 || true

GLSL_FILES=(
    ts_polyline.task.glsl ts_polyline.mesh.glsl ts_polyline.frag.glsl
    ts_polygon.task.glsl  ts_polygon.mesh.glsl  ts_polygon.frag.glsl
    ts_obstacle.task.glsl ts_obstacle.mesh.glsl ts_obstacle.frag.glsl
)

failed=0
for f in "${GLSL_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  missing source: $f"
        failed=$((failed+1))
        continue
    fi
    spv="${f%.glsl}.spv"
    if "$GLSLANG" --target-env vulkan1.3 --target-env spirv1.6 -V -o "$spv" "$f"; then
        :
    else
        failed=$((failed+1))
    fi
done

if [ $failed -ne 0 ]; then
    echo "FAILED: $failed shader(s) did not compile"
    exit 1
fi

echo "All shaders compiled. Embedding into C++ header..."
python3 "$SCRIPT_DIR/embed_spv.py"
echo "Done."
