# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual CUDA clipping math with a host compiler."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


def test_crossing_faces_keep_visible_area_winding_and_color(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the host clipping check")
    source = (
        Path(__file__).parents[1] / "ludus_renderer/_cpp/render/ludus_cuda.cu"
    ).read_text()

    def function(name):
        match = re.search(
            r"__device__\s+static[^;{}]*\b" + name + r"\([^;{}]*\)\s*\{", source
        )
        assert match is not None, name
        depth, end = 1, match.end()
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[match.start() : end]

    assert source.count("emit_clipped_subtriangle(") == 3
    near = re.search(r"constexpr float NEAR_DEPTH = [^;]+;", source).group()
    code = (
        """
#include <cassert>
#include <cmath>
#define __device__
#define __forceinline__ inline
struct float3 { float x, y, z; };
float3 make_float3(float x, float y, float z) { return {x,y,z}; }
"""
        + near
        + "\n"
        + "\n".join(
            function(name)
            for name in ("camera_depth", "pulled_to_near", "near_clipped_triangle")
        )
        + """
int main() {
    float pose[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    float3 triangle[3] = {{-1,0,-1}, {1,0,1}, {0,1,1}};
    float attributes[3] = {0,1,1}, colors[2][3];
    float3 output[2][3];
    int count = near_clipped_triangle(triangle, pose, 1, output, attributes, colors);
    assert(count == 2);
    float area = 0;
    for (int i=0; i<count; ++i) {
        const auto& p = output[i];
        float winding = (p[1].x-p[0].x)*(p[2].y-p[0].y) - (p[1].y-p[0].y)*(p[2].x-p[0].x);
        assert(winding > 0);
        area += winding / 2;
        for (int j=0; j<3; ++j) {
            assert(p[j].z > NEAR_DEPTH);
            assert(std::abs(colors[i][j] - (p[j].z+1)/2) < 1e-5);
        }
    }
    assert(area > .5 && area < 1);
    for (auto& p : triangle) p.z = -1;
    assert(near_clipped_triangle(triangle, pose, 1, output, attributes, colors) == 0);
    assert(near_clipped_triangle(triangle, pose, 0, output, attributes, colors) == 1);
    for (auto& p : triangle) p.z = 1;
    assert(near_clipped_triangle(triangle, pose, 1, output, nullptr, nullptr) == 1);
}
"""
    )
    path = tmp_path / "clip.cpp"
    path.write_text(code)
    executable = tmp_path / "clip"
    subprocess.run(
        [compiler, "-std=c++17", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
