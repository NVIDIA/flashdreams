// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//------------------------------------------------------------------------
// Ludus renderer.
//
// CUDA kernels generate geometry from world-space primitives, then feed
// triangles to CudaRaster (HPG 2011 software rasterizer).
//
// Pipeline (per camera):
//   1. Geometry kernels: world-space primitives → projected vertices + triangles
//   2. CudaRaster: vertices + triangles → triangle_id per pixel + depth
//   3. Fragment kernel: triangle_id → RGBA8 output
//------------------------------------------------------------------------

#include "ludus_cuda.h"
#include "ludus_types.h"
#include "../cudaraster/CudaRaster.hpp"
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(CALL) do { cudaError_t err = CALL; if (err != cudaSuccess) { fprintf(stderr, "CUDA error %d (%s) at %s:%d: %s\n", (int)err, cudaGetErrorString(err), __FILE__, __LINE__, #CALL); } } while(0)

//------------------------------------------------------------------------
// Device helpers: camera projection and Rodrigues rotation.
//------------------------------------------------------------------------

static __device__ __forceinline__ float3 rodrigues(float3 v, float3 r)
{
    float theta = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (theta < 1e-8f) return v;
    float inv_theta = 1.0f / theta;
    float3 k = make_float3(r.x * inv_theta, r.y * inv_theta, r.z * inv_theta);
    float c = cosf(theta);
    float s = sinf(theta);
    float dot_kv = k.x*v.x + k.y*v.y + k.z*v.z;
    float3 cross_kv = make_float3(
        k.y*v.z - k.z*v.y,
        k.z*v.x - k.x*v.z,
        k.x*v.y - k.y*v.x
    );
    return make_float3(
        v.x*c + cross_kv.x*s + k.x*dot_kv*(1.0f - c),
        v.y*c + cross_kv.y*s + k.y*dot_kv*(1.0f - c),
        v.z*c + cross_kv.z*s + k.z*dot_kv*(1.0f - c)
    );
}

// Camera projection: world point → clip-space float4 for CudaRaster.
// Returns (ndc_x, ndc_y, z_ndc, 1.0) — w=1 since projection is baked in.
static __device__ float4 ftheta_project(
    float3 world_pos,
    const float* __restrict__ pose,  // 16 floats, column-major 4x4
    const float* __restrict__ cam)   // 18 floats, FThetaCamera layout
{
    // Transform world → camera
    float3 cam_pt = make_float3(
        pose[0]*world_pos.x + pose[4]*world_pos.y + pose[8]*world_pos.z  + pose[12],
        pose[1]*world_pos.x + pose[5]*world_pos.y + pose[9]*world_pos.z  + pose[13],
        pose[2]*world_pos.x + pose[6]*world_pos.y + pose[10]*world_pos.z + pose[14]
    );
    float depth = cam_pt.z;
    float ray_norm = sqrtf(cam_pt.x*cam_pt.x + cam_pt.y*cam_pt.y + cam_pt.z*cam_pt.z);

    // Unpack camera intrinsics
    float cx = cam[0], cy = cam[1];
    float img_w = cam[2], img_h = cam[3];
    // cam[4..9] = poly[0..5]
    float max_ray_angle = cam[10];
    float max_distortion_val = cam[11];
    float max_distortion_dval = cam[12];
    float depth_max = cam[13];
    float ld_c = cam[14], ld_d = cam[15], ld_e = cam[16], ld_f = cam[17];

    // Orthographic cameras use a negative max-ray-angle marker and store
    // pixels-per-metre in the first two polynomial slots. The packed layout
    // stays ABI-compatible with FThetaCamera.
    if (max_ray_angle < 0.0f) {
        if (depth < 0.001f)
            return make_float4(10.0f, 10.0f, 1.0f, 1.0f);

        float px = cx + cam_pt.x * cam[4];
        float py = cy + cam_pt.y * cam[5];
        float x_ndc = 2.0f * px / img_w - 1.0f;
        float y_ndc = 1.0f - 2.0f * py / img_h;
        float z_ndc = fminf(
            fmaxf((depth / depth_max) * 2.0f - 1.0f, -1.0f), 1.0f);
        return make_float4(x_ndc, y_ndc, z_ndc, 1.0f);
    }

    if (ray_norm < 1e-6f)
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    const float HALF_PI = 1.5707963f;

    if (max_ray_angle <= HALF_PI && depth < 0.001f) {
        float pseudo_focal = cam[5]; // poly[1]
        float x_clip = cam_pt.x * pseudo_focal / (img_w * 0.5f);
        float y_clip = -cam_pt.y * pseudo_focal / (img_h * 0.5f);
        return make_float4(x_clip * 10.0f, y_clip * 10.0f, 1.0f, 1.0f);
    }

    float xy_norm = sqrtf(cam_pt.x*cam_pt.x + cam_pt.y*cam_pt.y);
    float cos_alpha = fminf(fmaxf(cam_pt.z / ray_norm, -1.0f), 1.0f);
    float alpha = acosf(cos_alpha);

    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a2 * a2;
    float a5 = a4 * alpha;

    float delta = cam[4] + cam[5]*alpha + cam[6]*a2 + cam[7]*a3 + cam[8]*a4 + cam[9]*a5;
    if (alpha > max_ray_angle)
        delta = max_distortion_val + (alpha - max_ray_angle) * max_distortion_dval;

    float scale = (xy_norm > 1e-6f) ? (delta / xy_norm) : 0.0f;
    float px_rel = scale * cam_pt.x;
    float py_rel = scale * cam_pt.y;

    float px_dist = ld_c * px_rel + ld_d * py_rel;
    float py_dist = ld_e * px_rel + ld_f * py_rel;

    float px = px_dist + cx;
    float py = py_dist + cy;

    float x_ndc = 2.0f * px / img_w - 1.0f;
    float y_ndc = 1.0f - 2.0f * py / img_h;

    float z_value;
    if (max_ray_angle > HALF_PI)
        z_value = (depth >= 0.0f) ? ray_norm : depth_max;
    else
        z_value = depth;
    float z_ndc = fminf(fmaxf((z_value / depth_max) * 2.0f - 1.0f, -1.0f), 1.0f);

    return make_float4(x_ndc, y_ndc, z_ndc, 1.0f);
}

static __device__ __forceinline__ uint32_t pack_rgba8(float r, float g, float b)
{
    uint32_t rb = (uint32_t)fminf(fmaxf(r * 255.0f + 0.5f, 0.0f), 255.0f);
    uint32_t gb = (uint32_t)fminf(fmaxf(g * 255.0f + 0.5f, 0.0f), 255.0f);
    uint32_t bb = (uint32_t)fminf(fmaxf(b * 255.0f + 0.5f, 0.0f), 255.0f);
    return rb | (gb << 8) | (bb << 16) | (0xFFu << 24);
}

static __device__ __forceinline__ bool geometry_reservation_fits(
    int vbase, int vertexCount,
    int triBase, int triangleCount,
    int* atomicVerts, int* atomicTris)
{
    if (vbase < 0 || triBase < 0 ||
        vertexCount > atomicVerts[1] - vbase ||
        triangleCount > atomicTris[1] - triBase) {
        atomicExch(&atomicVerts[2], 1);
        return false;
    }
    return true;
}

//------------------------------------------------------------------------
// Adaptive tessellation helpers.
//------------------------------------------------------------------------

static __device__ float estimate_edge_distortion_pixels(
    float3 v0, float3 v1,
    const float* __restrict__ poseData,
    const float* __restrict__ camData)
{
    float3 mid = make_float3(
        (v0.x + v1.x) * 0.5f,
        (v0.y + v1.y) * 0.5f,
        (v0.z + v1.z) * 0.5f);
    float4 p0 = ftheta_project(v0, poseData, camData);
    float4 p1 = ftheta_project(v1, poseData, camData);
    float4 pm = ftheta_project(mid, poseData, camData);
    float img_w = camData[2], img_h = camData[3];
    float sx0 = p0.x * img_w * 0.5f, sy0 = p0.y * img_h * 0.5f;
    float sx1 = p1.x * img_w * 0.5f, sy1 = p1.y * img_h * 0.5f;
    float sxm = pm.x * img_w * 0.5f, sym = pm.y * img_h * 0.5f;
    float ex = (sx0 + sx1) * 0.5f - sxm;
    float ey = (sy0 + sy1) * 0.5f - sym;
    return sqrtf(ex * ex + ey * ey);
}

// Polyline subdivision thresholds: 1x, 4x, 16x, 64x (up to level 4)
static __device__ __forceinline__ int compute_subdiv_level(
    float error, float threshold, int max_level)
{
    if (threshold <= 0.0f) return 0;
    int level = 0;
    if (error > threshold)         level = 1;
    if (error > threshold * 4.0f)  level = 2;
    if (error > threshold * 16.0f) level = 3;
    if (error > threshold * 64.0f) level = 4;
    return level < max_level ? level : max_level;
}

// Polygon/cube subdivision thresholds: 1x, 2x, 4x (more aggressive, up to max_level)
// Matches GL compute_subdivision_level()
static __device__ __forceinline__ int compute_subdiv_level_polygon(
    float error, float threshold, int max_level = 3)
{
    if (threshold <= 0.0f || error < threshold) return 0;
    int level = 1;
    if (error >= threshold * 2.0f) level = 2;
    if (error >= threshold * 4.0f) level = 3;
    return level < max_level ? level : max_level;
}

// Barycentric subdivision: vertex count = (s+1)(s+2)/2 where s = 2^level
static __device__ __forceinline__ int bary_vertex_count(int level)
{
    const int LUT[] = {3, 6, 15, 45};
    return LUT[level];
}

// Barycentric subdivision: triangle count = 4^level
static __device__ __forceinline__ int bary_triangle_count(int level)
{
    const int LUT[] = {1, 4, 16, 64};
    return LUT[level];
}

// Flat vertex index → barycentric UV
static __device__ float2 bary_vertex_uv(int idx, int level)
{
    int s = 1 << level;
    int row = 0, cumul = 0;
    for (int r = 0; r <= s; r++) {
        int row_sz = s + 1 - r;
        if (idx < cumul + row_sz) { row = r; break; }
        cumul += row_sz;
    }
    int col = idx - cumul;
    return make_float2((float)col / (float)s, (float)row / (float)s);
}

// Flat triangle index → 3 vertex indices within the bary grid
static __device__ int3 bary_triangle_indices(int tri_idx, int level)
{
    int s = 1 << level;
    int row = 0, cumul = 0;
    for (int r = 0; r < s; r++) {
        int tris_in_row = 2 * (s - r) - 1;
        if (tri_idx < cumul + tris_in_row) { row = r; break; }
        cumul += tris_in_row;
    }
    int local = tri_idx - cumul;
    int rs  = row * (s + 1) - row * (row - 1) / 2;
    int nrs = (row + 1) * (s + 1) - (row + 1) * row / 2;
    if (local % 2 == 0) {
        int col = local / 2;
        return make_int3(rs + col, rs + col + 1, nrs + col);
    } else {
        int col = (local - 1) / 2;
        return make_int3(rs + col + 1, nrs + col + 1, nrs + col);
    }
}

//------------------------------------------------------------------------
// Device helpers for timestamped pool processing.
// Binary search, color/width lookup — ports of GL task shader logic.
//------------------------------------------------------------------------

// Binary search: find largest index i such that timestamps[base + i] <= target.
// Returns -1 if no such index exists (all timestamps are > target or count==0).
static __device__ int binary_search_timestamps(
    const int64_t* __restrict__ timestamps,
    uint32_t base_offset, uint32_t count, int64_t target)
{
    if (count == 0) return -1;
    int left = 0, right = (int)count - 1, result = -1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (timestamps[base_offset + mid] <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return result;
}

// Default color palette matching GL get_default_prim_color() (imaginaire4 v3)
static __device__ uint32_t get_default_prim_color(uint32_t prim_type_id)
{
    switch (prim_type_id) {
        case  0: return pack_rgba8(253/255.f,  1/255.f, 232/255.f); // road_boundary
        case  1: return pack_rgba8( 98/255.f,183/255.f, 249/255.f); // lane_line
        case  2: return pack_rgba8(139/255.f, 93/255.f, 255/255.f); // crosswalk
        case  3: return pack_rgba8(255/255.f,100/255.f,   0/255.f); // static_obstacle
        case  4: return pack_rgba8(  0/255.f,255/255.f,   0/255.f); // ego_trajectory
        case  5: return pack_rgba8(255/255.f,100/255.f,   0/255.f); // obstacle
        case  6: return pack_rgba8(255/255.f,100/255.f,   0/255.f); // ego_obstacle
        case  7: return pack_rgba8(108/255.f,179/255.f,  59/255.f); // wait_line
        case  8: return pack_rgba8(183/255.f, 69/255.f, 177/255.f); // pole
        case  9: return pack_rgba8( 20/255.f,254/255.f, 185/255.f); // road_marking
        case 10: return pack_rgba8( 98/255.f,183/255.f, 249/255.f); // lane_boundary
        case 11: return pack_rgba8(100/255.f,100/255.f, 100/255.f); // traffic_light
        case 12: return pack_rgba8(  8/255.f,  2/255.f, 255/255.f); // traffic_sign
        case 13: return pack_rgba8( 80/255.f, 80/255.f, 120/255.f); // intersection
        case 14: return pack_rgba8( 60/255.f,120/255.f,  60/255.f); // road_island
        case 15: return pack_rgba8(120/255.f, 80/255.f,  80/255.f); // buffer_zone
        case 16: // lane_line_white_solid
        case 17: return pack_rgba8(255/255.f,255/255.f, 255/255.f); // lane_line_white_dashed
        case 18: // lane_line_yellow_solid
        case 19: return pack_rgba8(255/255.f,255/255.f,   0/255.f); // lane_line_yellow_dashed
        case 20: return pack_rgba8(255/255.f,255/255.f,   0/255.f); // dot_yellow
        case 21: return pack_rgba8(255/255.f,255/255.f, 255/255.f); // dot_white
        case 22: return pack_rgba8(  0/255.f,  0/255.f,   0/255.f); // BEV road surface
        default: return pack_rgba8(1.0f, 1.0f, 1.0f);              // default white
    }
}

static __device__ uint32_t get_prim_color_packed(uint32_t prim_type_id, const CudaRenderParams& p)
{
    if (p.colorPaletteSize > 0 && p.colorPalette && prim_type_id < (uint32_t)p.colorPaletteSize) {
        uint32_t c = p.colorPalette[prim_type_id];
        if ((c >> 24) != 0) return c;
    }
    return get_default_prim_color(prim_type_id);
}

static __device__ __forceinline__ float get_prim_width(uint32_t prim_type_id, const CudaRenderParams& p)
{
    bool is_bev = (p.cameraTypeId == 1);
    float base;
    // Compared unsigned: a pool's id travels in a signed header, so a negative
    // one arrives here as a very large value and a signed test would let it
    // through to index the table from outside.
    if (p.widthTableSize > 0 && p.widthTable && prim_type_id < (uint32_t)p.widthTableSize
        && p.widthTable[prim_type_id] > 0.0f) {
        base = p.widthTable[prim_type_id];
    } else if (prim_type_id == 4) {
        float def = is_bev ? 5.0f : 12.0f;
        float cust = is_bev ? p.widthEgoTrajBev : p.widthEgoTrajRegular;
        base = (cust > 0.0f) ? cust : def;
    } else if (prim_type_id == 8) {
        base = is_bev ? 3.0f : 5.0f;
    } else {
        float def = is_bev ? 4.0f : 7.0f;
        float cust = is_bev ? p.widthPolylineBev : p.widthPolylineRegular;
        base = (cust > 0.0f) ? cust : def;
    }
    float scale = (p.resolutionScale > 0.0f) ? p.resolutionScale : 1.0f;
    return base * scale;
}

// Where a line's two sides go, given its direction in pixels and its half
// width. In pixels the sides sit square; in NDC they step equally along both
// clip axes from the picture's width alone, so a line across the picture comes
// out narrower than one up it by the aspect ratio, which is what a caller
// matching a renderer that offsets in NDC wants. Both take the perpendicular
// to the same side, so the modes differ in width alone.
static __device__ __forceinline__ void get_line_offset(
    float dx, float dy, float half_width, float miter_scale,
    float img_w, float img_h, int width_in_ndc, float clip_w,
    float& ox, float& oy)
{
    if (width_in_ndc) {
        float ux = dx / img_w, uy = -dy / img_h;
        float ul = sqrtf(ux*ux + uy*uy);
        if (ul > 1e-9f) { ux /= ul; uy /= ul; }
        float ndc_hw = 2.0f * half_width * miter_scale / img_w;
        ox =  uy * ndc_hw * clip_w;
        oy = -ux * ndc_hw * clip_w;
    } else {
        ox = -dy * half_width * miter_scale * 2.0f / img_w * clip_w;
        oy = -dx * half_width * miter_scale * 2.0f / img_h * clip_w;
    }
}

static __device__ __forceinline__ float get_wireframe_width(const CudaRenderParams& p)
{
    float base = (p.widthWireframe > 0.0f) ? p.widthWireframe : 2.0f;
    float scale = (p.resolutionScale > 0.0f) ? p.resolutionScale : 1.0f;
    return base * scale;
}

// A point on a round cap or a dot, offset from its centre in whatever
// convention get_line_offset is drawing the line's body in. `sideways` and
// `along` are the components wanted, across the line and down it, so (1, 0)
// reproduces get_line_offset exactly and the cap meets the body's own side
// vertices, which the fan is stitched to and so cannot be turned without it.
static __device__ __forceinline__ void get_cap_offset(
    float dx, float dy, float half_width, float sideways, float along,
    float img_w, float img_h, int width_in_ndc, float clip_w,
    float& ox, float& oy)
{
    if (width_in_ndc) {
        float ux = dx / img_w, uy = -dy / img_h;
        float ul = sqrtf(ux*ux + uy*uy);
        if (ul > 1e-9f) { ux /= ul; uy /= ul; }
        float ndc_hw = 2.0f * half_width / img_w;
        ox = ( uy * sideways + ux * along) * ndc_hw * clip_w;
        oy = (-ux * sideways + uy * along) * ndc_hw * clip_w;
    } else {
        float rx = -dy * sideways + dx * along;
        float ry =  dx * sideways + dy * along;
        ox =  rx * half_width * 2.0f / img_w * clip_w;
        oy = -ry * half_width * 2.0f / img_h * clip_w;
    }
}

// An outline edge's offset. An edge is a line like any other, so it is laid
// out as a polyline of the same width would be, narrowing with distance -- and
// only while depth_scaling asks for it, since a caller that turns it off means
// it of everything it draws, outlines included.
static __device__ __forceinline__ void get_edge_offset(
    float4 ca, float4 cb, float wa, float wb,
    float img_w, float img_h, float width, int width_in_ndc,
    float depth_scaling, float& ox, float& oy)
{
    float dx = (cb.x / wb - ca.x / wa) * 0.5f * img_w;
    float dy = -(cb.y / wb - ca.y / wa) * 0.5f * img_h;
    float dl = sqrtf(dx*dx + dy*dy);
    if (dl > 0.001f) { dx /= dl; dy /= dl; } else { dx = 1.0f; dy = 0.0f; }

    float z_ndc = (ca.z / wa + cb.z / wb) * 0.5f;
    float depth_scale = (depth_scaling > 0.5f)
        ? fminf(fmaxf((1.0f - z_ndc) * 0.5f, 0.0f), 1.0f) : 1.0f;
    get_line_offset(dx, dy, width * 0.5f * depth_scale, 1.0f,
                    img_w, img_h, width_in_ndc, 1.0f, ox, oy);
}

static __device__ __forceinline__ bool is_dot_primitive(uint32_t prim_type_id)
{
    return prim_type_id == 20 || prim_type_id == 21;
}

//------------------------------------------------------------------------
// Geometry generation kernels.
// All kernels operate on a SINGLE camera. Host loops over cameras.
//------------------------------------------------------------------------

// Polygon: one thread per triangle of a polygon. All polygons in one launch.
// With adaptive tessellation: each thread may generate sub-triangles for its triangle.
__global__ void polygonGeometryKernel(
    const PolygonHeader* __restrict__ headers, int numPolygons,
    const Vertex* __restrict__ vertices,
    const Triangle* __restrict__ triangles,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    float tessThreshold, int maxTessLevel,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    int pgIdx = blockIdx.x;
    if (pgIdx >= numPolygons) return;

    const PolygonHeader& pg = headers[pgIdx];
    uint32_t color = pack_rgba8(pg.color[0], pg.color[1], pg.color[2]);

    for (int t = (int)threadIdx.x; t < (int)pg.triangle_count; t += (int)blockDim.x) {
        const Triangle& tri = triangles[pg.triangle_start + t];

        float3 wp[3];
        for (int vi = 0; vi < 3; vi++) {
            uint32_t li = tri.indices[vi];
            const Vertex& vtx = vertices[pg.vertex_start + li];
            wp[vi] = make_float3(vtx.position[0], vtx.position[1], vtx.position[2]);
        }

        int level = 0;
        if (tessThreshold > 0.0f) {
            float e01 = estimate_edge_distortion_pixels(wp[0], wp[1], poseData, camData);
            float e12 = estimate_edge_distortion_pixels(wp[1], wp[2], poseData, camData);
            float e20 = estimate_edge_distortion_pixels(wp[2], wp[0], poseData, camData);
            float emax = fmaxf(e01, fmaxf(e12, e20));
            level = compute_subdiv_level_polygon(emax, tessThreshold, maxTessLevel);
        }

        if (level == 0) {
            int vbase = atomicAdd(atomicVerts, 3);
            int triOut = atomicAdd(atomicTris, 1);
            if (!geometry_reservation_fits(vbase, 3, triOut, 1, atomicVerts, atomicTris))
                return;
            for (int vi = 0; vi < 3; vi++) {
                outVerts[vbase + vi] = ftheta_project(wp[vi], poseData, camData);
                outVertColors[vbase + vi] = color;
            }
            outIndices[triOut * 3 + 0] = vbase;
            outIndices[triOut * 3 + 1] = vbase + 1;
            outIndices[triOut * 3 + 2] = vbase + 2;
        } else {
            int nV = bary_vertex_count(level);
            int nT = bary_triangle_count(level);
            int vbase = atomicAdd(atomicVerts, nV);
            int triBase = atomicAdd(atomicTris, nT);
            if (!geometry_reservation_fits(vbase, nV, triBase, nT, atomicVerts, atomicTris))
                return;
            for (int v = 0; v < nV; v++) {
                float2 uv = bary_vertex_uv(v, level);
                float wb = 1.0f - uv.x - uv.y;
                float3 wp_sub = make_float3(
                    wb * wp[0].x + uv.x * wp[1].x + uv.y * wp[2].x,
                    wb * wp[0].y + uv.x * wp[1].y + uv.y * wp[2].y,
                    wb * wp[0].z + uv.x * wp[1].z + uv.y * wp[2].z);
                outVerts[vbase + v] = ftheta_project(wp_sub, poseData, camData);
                outVertColors[vbase + v] = color;
            }
            for (int ti = 0; ti < nT; ti++) {
                int3 idx = bary_triangle_indices(ti, level);
                outIndices[(triBase + ti) * 3 + 0] = vbase + idx.x;
                outIndices[(triBase + ti) * 3 + 1] = vbase + idx.y;
                outIndices[(triBase + ti) * 3 + 2] = vbase + idx.z;
            }
        }
    }
}

//------------------------------------------------------------------------
// Cube: one thread per face, one block per cube.

__device__ static const float3 CUBE_VERTS_D[8] = {
    {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f},
    {+0.5f, +0.5f, -0.5f}, {-0.5f, +0.5f, -0.5f},
    {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f},
    {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f}
};

__device__ static const int FACE_VERTS_D[6][4] = {
    {0, 3, 2, 1}, {4, 5, 6, 7},
    {0, 4, 7, 3}, {1, 2, 6, 5},
    {0, 1, 5, 4}, {3, 7, 6, 2}
};

__device__ static const float3 FACE_NORMALS_D[6] = {
    { 0,  0, -1}, { 0,  0, +1},
    {-1,  0,  0}, {+1,  0,  0},
    { 0, -1,  0}, { 0, +1,  0}
};

__device__ static const int CUBE_EDGES_D[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
};

// Each edge is adjacent to 2 faces
__device__ static const int EDGE_FACES_D[12][2] = {
    {0,4}, {0,3}, {0,5}, {0,2},
    {1,4}, {1,3}, {1,5}, {1,2},
    {2,4}, {3,4}, {3,5}, {2,5}
};

__device__ static float3 cube_cam_world(const float* __restrict__ poseData)
{
    return make_float3(
        -(poseData[0]*poseData[12] + poseData[1]*poseData[13] + poseData[2]*poseData[14]),
        -(poseData[4]*poseData[12] + poseData[5]*poseData[13] + poseData[6]*poseData[14]),
        -(poseData[8]*poseData[12] + poseData[9]*poseData[13] + poseData[10]*poseData[14])
    );
}

__device__ static uint32_t cube_face_mask(
    const Cube& cube, float3 tr, float3 sc, float3 rot, float3 cam_world)
{
    uint32_t mask = 0;
    for (int f = 0; f < 6; f++) {
        float3 n_local = FACE_NORMALS_D[f];
        float3 n_world = rodrigues(n_local, rot);
        float3 cl = make_float3(n_local.x * 0.5f * sc.x, n_local.y * 0.5f * sc.y, n_local.z * 0.5f * sc.z);
        float3 cw = rodrigues(cl, rot);
        cw.x += tr.x; cw.y += tr.y; cw.z += tr.z;
        float3 to_cam = make_float3(cam_world.x - cw.x, cam_world.y - cw.y, cam_world.z - cw.z);
        if (n_world.x*to_cam.x + n_world.y*to_cam.y + n_world.z*to_cam.z > 0.0f)
            mask |= (1u << f);
    }
    return mask;
}

// How near the camera a drawn point may sit, which is the reference's own
// near plane.
constexpr float NEAR_DEPTH = 1e-3f;

// How far in front of the camera a world point sits, negative behind it.
__device__ static __forceinline__ float camera_depth(
    float3 world, const float* __restrict__ poseData)
{
    return poseData[2]*world.x + poseData[6]*world.y + poseData[10]*world.z + poseData[14];
}

// The behind-camera tests, defined with the rest of them further down and
// declared here because the immediate-mode cube kernels come first in the file.
__device__ static bool cube_in_front_euler(
    float3 tr, float3 rot, float3 sc, const float* __restrict__ poseData, int culling);
__device__ static bool face_in_front(
    const float3* corners, const float* __restrict__ poseData, int culling);
__device__ static int near_clipped_triangle(
    const float3* wp, const float* __restrict__ poseData, int culling, float3 out[2][3]);
__device__ static void clip_to_near(
    float3& from, float3& to, const float* __restrict__ poseData, int culling);

// Where a point on a cube sits along the cube's own x, which is what a face's
// colour is blended by. A corner reads this off the unit cube it came from; a
// corner the near-plane cut inserted has none, so it is recovered by undoing
// the cube's rotation and scale.
__device__ static __forceinline__ float cube_gradient_at(
    float3 world, float3 tr, float3 rot, float3 sc)
{
    float3 local = rodrigues(
        make_float3(world.x - tr.x, world.y - tr.y, world.z - tr.z),
        make_float3(-rot.x, -rot.y, -rot.z));
    return (sc.x != 0.0f) ? (local.x / sc.x + 0.5f) : 0.5f;
}

__global__ void cubeGeometryKernel(
    const Cube* __restrict__ cubes, int numCubes,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    int cullBehind,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    int cubeIdx = blockIdx.x;
    int faceIdx = threadIdx.x;
    if (cubeIdx >= numCubes || faceIdx >= 6) return;

    const Cube& cube = cubes[cubeIdx];
    float3 tr = make_float3(cube.translation[0], cube.translation[1], cube.translation[2]);
    float3 sc = make_float3(cube.scale[0], cube.scale[1], cube.scale[2]);
    float3 rot = make_float3(cube.rotation[0], cube.rotation[1], cube.rotation[2]);
    float3 fc = make_float3(cube.front_color[0], cube.front_color[1], cube.front_color[2]);
    float3 bc = make_float3(cube.back_color[0], cube.back_color[1], cube.back_color[2]);

    if (!cube_in_front_euler(tr, rot, sc, poseData, cullBehind))
        return;

    float3 cw = cube_cam_world(poseData);

    // Per-face backface culling
    float3 n_local = FACE_NORMALS_D[faceIdx];
    float3 n_world = rodrigues(n_local, rot);
    float3 fc_local = make_float3(n_local.x * 0.5f * sc.x, n_local.y * 0.5f * sc.y, n_local.z * 0.5f * sc.z);
    float3 fc_world = rodrigues(fc_local, rot);
    fc_world.x += tr.x; fc_world.y += tr.y; fc_world.z += tr.z;
    float3 to_cam = make_float3(cw.x - fc_world.x, cw.y - fc_world.y, cw.z - fc_world.z);
    if (n_world.x*to_cam.x + n_world.y*to_cam.y + n_world.z*to_cam.z <= 0.0f)
        return;

    float3 corners[4];
    for (int i = 0; i < 4; i++) {
        float3 lv = CUBE_VERTS_D[FACE_VERTS_D[faceIdx][i]];
        float3 sv = make_float3(lv.x * sc.x, lv.y * sc.y, lv.z * sc.z);
        corners[i] = rodrigues(sv, rot);
        corners[i].x += tr.x; corners[i].y += tr.y; corners[i].z += tr.z;
    }
    if (!face_in_front(corners, poseData, cullBehind))
        return;

    // No tessellation here to cut a crossing face along, so each of its two
    // triangles is cut at the crossings themselves, as the polygon faces are.
    // A face clear of the plane keeps the cheaper four-corner path below.
    bool crossing = false;
    if (cullBehind) {
        for (int i = 0; i < 4; i++)
            crossing |= camera_depth(corners[i], poseData) <= NEAR_DEPTH;
    }
    if (crossing) {
        uint32_t flags;
        memcpy(&flags, &cube._pad0, sizeof(uint32_t));
        for (int half = 0; half < 2; half++) {
            float3 wp[3] = { corners[0],
                             corners[(half == 0) ? 1 : 2],
                             corners[(half == 0) ? 2 : 3] };
            float3 kept[2][3];
            int pieces = near_clipped_triangle(wp, poseData, cullBehind, kept);
            for (int piece = 0; piece < pieces; piece++) {
                int vb = atomicAdd(atomicVerts, 3);
                int tb = atomicAdd(atomicTris, 1);
                // Carry on rather than give up on the rest, as the dots do:
                // the retry sizes its buffers from these counters.
                if (!geometry_reservation_fits(vb, 3, tb, 1, atomicVerts, atomicTris))
                    continue;
                for (int vi = 0; vi < 3; vi++) {
                    float3 wv = kept[piece][vi];
                    float t = (flags & 2u) ? ((faceIdx == 3) ? 1.0f : 0.0f)
                                           : cube_gradient_at(wv, tr, rot, sc);
                    outVerts[vb + vi] = ftheta_project(wv, poseData, camData);
                    outVertColors[vb + vi] = pack_rgba8(
                        bc.x + t * (fc.x - bc.x),
                        bc.y + t * (fc.y - bc.y),
                        bc.z + t * (fc.z - bc.z));
                }
                outIndices[tb * 3 + 0] = vb;
                outIndices[tb * 3 + 1] = vb + 1;
                outIndices[tb * 3 + 2] = vb + 2;
            }
        }
        return;
    }

    int vbase = atomicAdd(atomicVerts, 4);
    int triBase = atomicAdd(atomicTris, 2);
    if (!geometry_reservation_fits(vbase, 4, triBase, 2, atomicVerts, atomicTris))
        return;

    for (int i = 0; i < 4; i++) {
        int vi = FACE_VERTS_D[faceIdx][i];
        float3 lv = CUBE_VERTS_D[vi];

        // Per-vertex gradient: t = local_vert.x + 0.5 (0 at back, 1 at front in FLU)
        // CUBE_FLAG_FLAT_FACES holds each face to one colour instead, the front
        // face taking the front colour and the other five the back, so a colour
        // meant for the front alone does not carry round the sides.
        uint32_t face_flags;
        memcpy(&face_flags, &cube._pad0, sizeof(uint32_t));
        float t = (face_flags & 2u) ? ((faceIdx == 3) ? 1.0f : 0.0f) : (lv.x + 0.5f);
        float cr = bc.x + t * (fc.x - bc.x);
        float cg = bc.y + t * (fc.y - bc.y);
        float cb = bc.z + t * (fc.z - bc.z);

        outVerts[vbase + i] = ftheta_project(corners[i], poseData, camData);
        outVertColors[vbase + i] = pack_rgba8(cr, cg, cb);
    }

    outIndices[triBase * 3 + 0] = vbase;
    outIndices[triBase * 3 + 1] = vbase + 1;
    outIndices[triBase * 3 + 2] = vbase + 2;
    outIndices[(triBase+1) * 3 + 0] = vbase;
    outIndices[(triBase+1) * 3 + 1] = vbase + 2;
    outIndices[(triBase+1) * 3 + 2] = vbase + 3;
}

//------------------------------------------------------------------------
// Cube wireframe: one thread per edge, one block per cube.
// Renders visible edges as thin screen-space quads with depth bias.

__global__ void cubeWireframeKernel(
    const Cube* __restrict__ cubes, int numCubes,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    float wireframeWidth, int widthInNdc, float depthScaling, int cullBehind,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    int cubeIdx = blockIdx.x;
    int edgeIdx = threadIdx.x;
    if (cubeIdx >= numCubes || edgeIdx >= 12) return;

    const Cube& cube = cubes[cubeIdx];

    // _pad0 carries render_flags; only emit wireframe if CUBE_FLAG_WIREFRAME (bit 0) is set
    uint32_t render_flags;
    memcpy(&render_flags, &cube._pad0, sizeof(uint32_t));
    if ((render_flags & 1u) == 0) return;

    float3 tr = make_float3(cube.translation[0], cube.translation[1], cube.translation[2]);
    float3 sc = make_float3(cube.scale[0], cube.scale[1], cube.scale[2]);
    float3 rot = make_float3(cube.rotation[0], cube.rotation[1], cube.rotation[2]);

    if (!cube_in_front_euler(tr, rot, sc, poseData, cullBehind))
        return;

    float3 cw = cube_cam_world(poseData);
    uint32_t fmask = cube_face_mask(cube, tr, sc, rot, cw);

    int f0 = EDGE_FACES_D[edgeIdx][0], f1 = EDGE_FACES_D[edgeIdx][1];
    if (((fmask >> f0) & 1) == 0 && ((fmask >> f1) & 1) == 0)
        return;

    int vi0 = CUBE_EDGES_D[edgeIdx][0], vi1 = CUBE_EDGES_D[edgeIdx][1];
    float3 lv0 = CUBE_VERTS_D[vi0], lv1 = CUBE_VERTS_D[vi1];
    float3 sv0 = make_float3(lv0.x*sc.x, lv0.y*sc.y, lv0.z*sc.z);
    float3 sv1 = make_float3(lv1.x*sc.x, lv1.y*sc.y, lv1.z*sc.z);
    float3 wv0 = rodrigues(sv0, rot); wv0.x += tr.x; wv0.y += tr.y; wv0.z += tr.z;
    float3 wv1 = rodrigues(sv1, rot); wv1.x += tr.x; wv1.y += tr.y; wv1.z += tr.z;
    clip_to_near(wv0, wv1, poseData, cullBehind);

    float4 clip0 = ftheta_project(wv0, poseData, camData);
    float4 clip1 = ftheta_project(wv1, poseData, camData);

    float img_w = camData[2], img_h = camData[3];
    float w0 = fmaxf(fabsf(clip0.w), 0.001f);
    float w1 = fmaxf(fabsf(clip1.w), 0.001f);
    float ox, oy;
    get_edge_offset(clip0, clip1, w0, w1, img_w, img_h,
                    wireframeWidth, widthInNdc, depthScaling, ox, oy);

    float z_bias0 = -0.001f * clip0.w;
    float z_bias1 = -0.001f * clip1.w;

    int vbase = atomicAdd(atomicVerts, 4);
    int triBase = atomicAdd(atomicTris, 2);
    if (!geometry_reservation_fits(vbase, 4, triBase, 2, atomicVerts, atomicTris))
        return;

    outVerts[vbase + 0] = make_float4(clip0.x - ox*clip0.w, clip0.y - oy*clip0.w, clip0.z + z_bias0, clip0.w);
    outVerts[vbase + 1] = make_float4(clip0.x + ox*clip0.w, clip0.y + oy*clip0.w, clip0.z + z_bias0, clip0.w);
    outVerts[vbase + 2] = make_float4(clip1.x + ox*clip1.w, clip1.y + oy*clip1.w, clip1.z + z_bias1, clip1.w);
    outVerts[vbase + 3] = make_float4(clip1.x - ox*clip1.w, clip1.y - oy*clip1.w, clip1.z + z_bias1, clip1.w);

    outIndices[triBase * 3 + 0] = vbase;
    outIndices[triBase * 3 + 1] = vbase + 1;
    outIndices[triBase * 3 + 2] = vbase + 2;
    outIndices[(triBase+1) * 3 + 0] = vbase;
    outIndices[(triBase+1) * 3 + 1] = vbase + 2;
    outIndices[(triBase+1) * 3 + 2] = vbase + 3;

    uint32_t edgeColor = pack_rgba8(0.784f, 0.784f, 0.784f);
    for (int i = 0; i < 4; i++)
        outVertColors[vbase + i] = edgeColor;
}

//------------------------------------------------------------------------
// Quaternion math (for timestamped cube pool interpolation on GPU).
//------------------------------------------------------------------------

static __device__ __forceinline__ float4 quat_slerp_d(float4 q0, float4 q1, float t)
{
    float d = q0.x*q1.x + q0.y*q1.y + q0.z*q1.z + q0.w*q1.w;
    if (d < 0.0f) { q1.x = -q1.x; q1.y = -q1.y; q1.z = -q1.z; q1.w = -q1.w; d = -d; }
    if (d > 0.9995f) {
        float4 r = make_float4(q0.x*(1-t)+q1.x*t, q0.y*(1-t)+q1.y*t, q0.z*(1-t)+q1.z*t, q0.w*(1-t)+q1.w*t);
        float len = sqrtf(r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w);
        float inv = 1.0f / fmaxf(len, 1e-10f);
        return make_float4(r.x*inv, r.y*inv, r.z*inv, r.w*inv);
    }
    float theta0 = acosf(fminf(fmaxf(d, -1.0f), 1.0f));
    float theta = theta0 * t;
    float st = sinf(theta), st0 = sinf(theta0);
    float s0 = cosf(theta) - d * st / st0;
    float s1 = st / st0;
    float4 r = make_float4(s0*q0.x+s1*q1.x, s0*q0.y+s1*q1.y, s0*q0.z+s1*q1.z, s0*q0.w+s1*q1.w);
    float len = sqrtf(r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w);
    float inv = 1.0f / fmaxf(len, 1e-10f);
    return make_float4(r.x*inv, r.y*inv, r.z*inv, r.w*inv);
}

// Rotate vector v by quaternion q (x,y,z,w)
static __device__ __forceinline__ float3 quat_rotate_d(float4 q, float3 v)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x+x, y2 = y+y, z2 = z+z;
    float xx = x*x2, xy = x*y2, xz = x*z2;
    float yy = y*y2, yz = y*z2, zz = z*z2;
    float wx = w*x2, wy = w*y2, wz = w*z2;
    return make_float3(
        (1-(yy+zz))*v.x + (xy-wz)*v.y + (xz+wy)*v.z,
        (xy+wz)*v.x + (1-(xx+zz))*v.y + (yz-wx)*v.z,
        (xz-wy)*v.x + (yz+wx)*v.y + (1-(xx+yy))*v.z
    );
}

static __device__ __forceinline__ bool point_inside_cube_d(
    float3 point, float3 tr, float4 qr, float3 sc)
{
    float3 delta = make_float3(point.x - tr.x, point.y - tr.y, point.z - tr.z);
    float4 inverse = make_float4(-qr.x, -qr.y, -qr.z, qr.w);
    float3 local = quat_rotate_d(inverse, delta);
    return fabsf(local.x) <= 0.5f * fabsf(sc.x)
        && fabsf(local.y) <= 0.5f * fabsf(sc.y)
        && fabsf(local.z) <= 0.5f * fabsf(sc.z);
}

// Whether the plane through the camera has all of a cube behind it.
//
// ftheta_project pushes a point behind the camera well off centre rather than
// dropping it, so a box across the camera's axis is drawn as a face spanning
// the frame. Opt-in, since a fisheye seeing past that plane loses a little of
// its own field to the test.
__device__ static bool cube_in_front(
    float3 tr, float4 qr, float3 sc, const float* __restrict__ poseData, int culling)
{
    if (!culling)
        return true;
    for (int corner = 0; corner < 8; corner++) {
        float3 local = CUBE_VERTS_D[corner];
        float3 world = quat_rotate_d(
            qr, make_float3(local.x * sc.x, local.y * sc.y, local.z * sc.z));
        world.x += tr.x; world.y += tr.y; world.z += tr.z;
        if (camera_depth(world, poseData) > 0.0f)
            return true;
    }
    return false;
}

// The same test for a cube whose rotation arrives as a Rodrigues vector, which
// is how the immediate-mode path carries it rather than as a quaternion.
__device__ static bool cube_in_front_euler(
    float3 tr, float3 rot, float3 sc, const float* __restrict__ poseData, int culling)
{
    if (!culling)
        return true;
    for (int corner = 0; corner < 8; corner++) {
        float3 local = CUBE_VERTS_D[corner];
        float3 world = rodrigues(
            make_float3(local.x * sc.x, local.y * sc.y, local.z * sc.z), rot);
        world.x += tr.x; world.y += tr.y; world.z += tr.z;
        if (camera_depth(world, poseData) > 0.0f)
            return true;
    }
    return false;
}

// The same question of one face. One corner in front keeps it; what of it is
// really in front is settled per piece by subtri_in_front, or for an
// untessellated face by face_wholly_in_front.
__device__ static bool face_in_front(
    const float3* corners, const float* __restrict__ poseData, int culling)
{
    if (!culling)
        return true;
    for (int corner = 0; corner < 4; corner++) {
        if (camera_depth(corners[corner], poseData) > 0.0f)
            return true;
    }
    return false;
}

// An edge cut back to the near plane, as the reference cuts its own. An end
// behind the camera comes forward to the crossing; an edge with both ends
// behind collapses to a point, its room in the buffer being already spoken for.
__device__ static void clip_to_near(
    float3& from, float3& to, const float* __restrict__ poseData, int culling)
{
    if (!culling)
        return;
    float at_from = camera_depth(from, poseData), at_to = camera_depth(to, poseData);
    if (at_from > NEAR_DEPTH && at_to > NEAR_DEPTH)
        return;
    if (at_from <= NEAR_DEPTH && at_to <= NEAR_DEPTH) {
        to = from;
        return;
    }
    float across = (NEAR_DEPTH - at_from) / (at_to - at_from);
    float3 crossing = make_float3(
        from.x + across * (to.x - from.x),
        from.y + across * (to.y - from.y),
        from.z + across * (to.z - from.z));
    if (at_from <= NEAR_DEPTH)
        from = crossing;
    else
        to = crossing;
}

// The stricter question, for a face with no tessellation to cut down: all four
// corners in front, or none of it is drawn.
__device__ static bool face_wholly_in_front(
    const float3* corners, const float* __restrict__ poseData, int culling)
{
    if (!culling)
        return true;
    for (int corner = 0; corner < 4; corner++) {
        if (camera_depth(corners[corner], poseData) <= NEAR_DEPTH)
            return false;
    }
    return true;
}

// Four pieces to an edge, which cuts a crossing face close enough to the plane.
static const int CROSSING_SUBDIV = 2;

// The tessellation a face needs to be cut at the near plane, which is not the
// caller's to decide: a face with no seams has nothing to cut, so one corner
// behind the camera would take all of it, however much was still in view. A
// face clear of the plane keeps whatever tessellation it was given.
__device__ static __forceinline__ int subdiv_for_crossing(
    const float3* corners, const float* __restrict__ poseData,
    int culling, int subdiv)
{
    if (face_wholly_in_front(corners, poseData, culling))
        return subdiv;
    return (subdiv > CROSSING_SUBDIV) ? subdiv : CROSSING_SUBDIV;
}

// Whether one piece of a face stands wholly in front of the near plane, which
// cuts the face down to what was ever visible. Depth is affine in world
// position, so a piece's depth is the barycentric blend of the face's corner
// depths and nothing need be projected to judge it. The cut follows the
// tessellation, so it falls up to CROSSING_SUBDIV shy of the plane itself.
__device__ static __forceinline__ bool subtri_in_front(
    int3 idx, int subdiv, float d0, float d1, float d2, int culling)
{
    if (!culling)
        return true;
    int at[3] = { idx.x, idx.y, idx.z };
    for (int i = 0; i < 3; i++) {
        float2 uv = bary_vertex_uv(at[i], subdiv);
        if ((1.0f - uv.x - uv.y) * d0 + uv.x * d1 + uv.y * d2 <= NEAR_DEPTH)
            return false;
    }
    return true;
}

// A point behind the camera brought onto the near plane, along the line from a
// point in front of it. A ribbon's end cannot be dropped the way a cube's edge
// can, since the strip would close over the gap, so it is moved onto the plane
// as the reference moves its own.
__device__ static float3 pulled_to_near(
    float3 anchor, float3 point, const float* __restrict__ poseData)
{
    float at_anchor = camera_depth(anchor, poseData);
    float span = camera_depth(point, poseData) - at_anchor;
    if (span >= 0.0f)
        return point;
    // A shade in front of the plane rather than on it, since the projection
    // takes anything under NEAR_DEPTH for behind and a crossing solved for in
    // floats can land the wrong side of the depth it solved for.
    float across = fminf(fmaxf((NEAR_DEPTH - at_anchor) / span * 0.999f, 0.0f), 1.0f);
    return make_float3(
        anchor.x + across * (point.x - anchor.x),
        anchor.y + across * (point.y - anchor.y),
        anchor.z + across * (point.z - anchor.z));
}

// A triangle cut at the near plane, as none, one or two triangles. The corners
// are walked in order with crossings inserted where the edge changes side, so
// what comes out is wound as what went in, and one plane cuts a triangle into
// at most a quad -- hence two.
__device__ static int near_clipped_triangle(
    const float3* wp, const float* __restrict__ poseData, int culling, float3 out[2][3])
{
    if (culling) {
        float3 kept[4];
        int n = 0;
        for (int i = 0; i < 3 && n < 4; i++) {
            float3 here = wp[i], next = wp[(i + 1) % 3];
            bool here_in = camera_depth(here, poseData) > NEAR_DEPTH;
            bool next_in = camera_depth(next, poseData) > NEAR_DEPTH;
            if (here_in)
                kept[n++] = here;
            if (here_in != next_in && n < 4)
                kept[n++] = here_in ? pulled_to_near(here, next, poseData)
                                    : pulled_to_near(next, here, poseData);
        }
        if (n < 3)
            return 0;
        out[0][0] = kept[0]; out[0][1] = kept[1]; out[0][2] = kept[2];
        if (n == 3)
            return 1;
        out[1][0] = kept[0]; out[1][1] = kept[2]; out[1][2] = kept[3];
        return 2;
    }

    out[0][0] = wp[0]; out[0][1] = wp[1]; out[0][2] = wp[2];
    return 1;
}

//------------------------------------------------------------------------
// Fused timestamped cube pool kernel: geometry + wireframe in one launch.
// 256 threads/block = 8 warps. Each warp handles one cube.
// Lane 0: binary search + slerp, broadcast via __shfl_sync.
// Lanes 0-5: face geometry (with backface culling).
// Lanes 6-17: wireframe edges (conditional on renderFlags & 1).
// Lanes 18-31: idle (still cheap — full warp occupancy, no divergence cost).

static __device__ void cubePoolInterpolate(
    const int64_t* __restrict__ trackTs,
    const int32_t* __restrict__ prefixSum,
    const float* __restrict__ poolTrans,
    const float* __restrict__ poolQuat,
    int cubeIdx, int64_t queryTs, int maxExtrapUs,
    float& tx, float& ty, float& tz,
    float& qx, float& qy, float& qz, float& qw,
    bool& visible)
{
    visible = false;
    int tStart = (cubeIdx > 0) ? prefixSum[cubeIdx - 1] : 0;
    int tEnd = prefixSum[cubeIdx];
    int tLen = tEnd - tStart;
    if (tLen < 1) return;

    int64_t firstTs = trackTs[tStart];
    int64_t lastTs  = trackTs[tStart + tLen - 1];
    bool extB = queryTs < firstTs, extA = queryTs > lastTs;
    if (extB && ((firstTs - queryTs) > maxExtrapUs || tLen < 2)) return;
    if (extA && ((queryTs - lastTs) > maxExtrapUs || tLen < 2)) return;

    int idx0 = 0, idx1 = 0;
    float alpha = 0.0f;

    if (tLen == 1) { /* idx0 = idx1 = 0, alpha = 0 */ }
    else if (extB) {
        idx1 = 1;
        int64_t t0 = trackTs[tStart], t1 = trackTs[tStart + 1];
        alpha = (t1 > t0) ? (float)(queryTs - t0) / (float)(t1 - t0) : 0.0f;
    } else if (extA) {
        idx0 = tLen - 2; idx1 = tLen - 1;
        int64_t t0 = trackTs[tStart + idx0], t1 = trackTs[tStart + idx1];
        alpha = (t1 > t0) ? (float)(queryTs - t0) / (float)(t1 - t0) : 1.0f;
    } else {
        int left = 0, right = tLen - 2;
        while (left <= right) {
            int mid = (left + right) / 2;
            int64_t t0 = trackTs[tStart + mid], t1 = trackTs[tStart + mid + 1];
            if (t0 <= queryTs && queryTs <= t1) { idx0 = mid; break; }
            else if (queryTs < t0) right = mid - 1;
            else { left = mid + 1; idx0 = mid + 1; }
        }
        idx1 = (idx0 + 1 < tLen) ? idx0 + 1 : idx0;
        int64_t t0 = trackTs[tStart + idx0], t1 = trackTs[tStart + idx1];
        alpha = (t1 > t0) ? (float)(queryTs - t0) / (float)(t1 - t0) : 0.0f;
    }

    int pi0 = tStart + idx0, pi1 = tStart + idx1;
    tx = poolTrans[pi0*3+0]*(1-alpha) + poolTrans[pi1*3+0]*alpha;
    ty = poolTrans[pi0*3+1]*(1-alpha) + poolTrans[pi1*3+1]*alpha;
    tz = poolTrans[pi0*3+2]*(1-alpha) + poolTrans[pi1*3+2]*alpha;

    float ac = fminf(fmaxf(alpha, 0.0f), 1.0f);
    float4 q0v = make_float4(poolQuat[pi0*4], poolQuat[pi0*4+1], poolQuat[pi0*4+2], poolQuat[pi0*4+3]);
    float4 q1v = make_float4(poolQuat[pi1*4], poolQuat[pi1*4+1], poolQuat[pi1*4+2], poolQuat[pi1*4+3]);
    float4 qr = quat_slerp_d(q0v, q1v, ac);
    qx = qr.x; qy = qr.y; qz = qr.z; qw = qr.w;
    visible = true;
}

#define CUBES_PER_BLOCK 8
#define CUBE_POOL_BLOCK_SIZE (CUBES_PER_BLOCK * 32)

__global__ void cubePoolFusedKernel(
    const int64_t* __restrict__ trackTs,
    const int32_t* __restrict__ prefixSum,
    const float* __restrict__ poolTrans,
    const float* __restrict__ poolQuat,
    const float* __restrict__ poolScales,
    const float* __restrict__ poolColors,
    int numCubes, int64_t queryTs, int maxExtrapUs,
    uint32_t renderFlags,
    float wireframeWidth, int widthInNdc, float depthScaling, int cullBehind,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    float tessThreshold, int maxTessLevel,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    int warpInBlock = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int cubeIdx = blockIdx.x * CUBES_PER_BLOCK + warpInBlock;
    if (cubeIdx >= numCubes) return;

    // Lane 0: interpolate + compute subdivision level, then broadcast
    float tx = 0, ty = 0, tz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
    bool visible = false;
    int subdiv = 0;
    if (lane == 0) {
        cubePoolInterpolate(trackTs, prefixSum, poolTrans, poolQuat,
                            cubeIdx, queryTs, maxExtrapUs,
                            tx, ty, tz, qx, qy, qz, qw, visible);
        if (visible && tessThreshold > 0.0f) {
            float3 tr_l = make_float3(tx, ty, tz);
            float4 qr_l = make_float4(qx, qy, qz, qw);
            float3 sc_l = make_float3(poolScales[cubeIdx*3], poolScales[cubeIdx*3+1], poolScales[cubeIdx*3+2]);
            float emax = 0.0f;
            for (int e = 0; e < 12; e++) {
                float3 lv0 = CUBE_VERTS_D[CUBE_EDGES_D[e][0]];
                float3 lv1 = CUBE_VERTS_D[CUBE_EDGES_D[e][1]];
                float3 sv0 = make_float3(lv0.x*sc_l.x, lv0.y*sc_l.y, lv0.z*sc_l.z);
                float3 sv1 = make_float3(lv1.x*sc_l.x, lv1.y*sc_l.y, lv1.z*sc_l.z);
                float3 wv0 = quat_rotate_d(qr_l, sv0);
                wv0.x += tr_l.x; wv0.y += tr_l.y; wv0.z += tr_l.z;
                float3 wv1 = quat_rotate_d(qr_l, sv1);
                wv1.x += tr_l.x; wv1.y += tr_l.y; wv1.z += tr_l.z;
                float err = estimate_edge_distortion_pixels(wv0, wv1, poseData, camData);
                if (err > emax) emax = err;
            }
            subdiv = compute_subdiv_level_polygon(emax, tessThreshold, maxTessLevel);
        }
    }

    int vis_i = visible ? 1 : 0;
    vis_i = __shfl_sync(FULL_MASK, vis_i, 0);
    if (!vis_i) return;
    tx = __shfl_sync(FULL_MASK, tx, 0);
    ty = __shfl_sync(FULL_MASK, ty, 0);
    tz = __shfl_sync(FULL_MASK, tz, 0);
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);
    qz = __shfl_sync(FULL_MASK, qz, 0);
    qw = __shfl_sync(FULL_MASK, qw, 0);
    subdiv = __shfl_sync(FULL_MASK, subdiv, 0);

    float3 tr = make_float3(tx, ty, tz);
    float4 qr = make_float4(qx, qy, qz, qw);
    float3 sc = make_float3(poolScales[cubeIdx*3], poolScales[cubeIdx*3+1], poolScales[cubeIdx*3+2]);
    if (!cube_in_front(tr, qr, sc, poseData, cullBehind))
        return;

    float3 cw = cube_cam_world(poseData);
    bool cameraInside = point_inside_cube_d(cw, tr, qr, sc);

    // --- Lanes 0-5: face geometry (with barycentric tessellation) ---
    if (lane < 6) {
        int faceIdx = lane;
        float3 n_local = FACE_NORMALS_D[faceIdx];
        float3 n_world = quat_rotate_d(qr, n_local);
        float3 fc_local = make_float3(n_local.x*0.5f*sc.x, n_local.y*0.5f*sc.y, n_local.z*0.5f*sc.z);
        float3 fc_world = quat_rotate_d(qr, fc_local);
        fc_world.x += tr.x; fc_world.y += tr.y; fc_world.z += tr.z;
        float3 to_cam = make_float3(cw.x - fc_world.x, cw.y - fc_world.y, cw.z - fc_world.z);
        if (cameraInside || n_world.x*to_cam.x + n_world.y*to_cam.y + n_world.z*to_cam.z > 0.0f) {
            float3 fc_col = make_float3(poolColors[cubeIdx*6], poolColors[cubeIdx*6+1], poolColors[cubeIdx*6+2]);
            float3 bc_col = make_float3(poolColors[cubeIdx*6+3], poolColors[cubeIdx*6+4], poolColors[cubeIdx*6+5]);

            // Face = quad with 4 corners → 2 triangles
            float3 corners[4];
            float corner_t[4];
            for (int i = 0; i < 4; i++) {
                int vi = FACE_VERTS_D[faceIdx][i];
                float3 lv = CUBE_VERTS_D[vi];
                // CUBE_FLAG_FLAT_FACES holds a face to one colour, the front
                // face to the front colour and the other five to the back.
                corner_t[i] = (renderFlags & 2u)
                    ? ((faceIdx == 3) ? 1.0f : 0.0f)
                    : (lv.x + 0.5f);
                float3 sv = make_float3(lv.x*sc.x, lv.y*sc.y, lv.z*sc.z);
                corners[i] = quat_rotate_d(qr, sv);
                corners[i].x += tr.x; corners[i].y += tr.y; corners[i].z += tr.z;
            }
            // Wholly behind drops the face here; a face across the plane is
            // cut down to its pieces in front when its triangles are written,
            // which needs seams to cut along whether or not any were asked for.
            if (!face_in_front(corners, poseData, cullBehind))
                return;
            int faceSubdiv = subdiv_for_crossing(corners, poseData, cullBehind, subdiv);

            // 2 triangles per face: (c0,c1,c2) and (c0,c2,c3)
            int nV = bary_vertex_count(faceSubdiv);
            int nT = bary_triangle_count(faceSubdiv);
            int totalV = nV * 2;
            int totalT = nT * 2;
            int vbase = atomicAdd(atomicVerts, totalV);
            int triBase = atomicAdd(atomicTris, totalT);
            if (!geometry_reservation_fits(vbase, totalV, triBase, totalT, atomicVerts, atomicTris))
                return;

            for (int half = 0; half < 2; half++) {
                float3 v0 = corners[0];
                float3 v1 = (half == 0) ? corners[1] : corners[2];
                float3 v2 = (half == 0) ? corners[2] : corners[3];
                float t0 = corner_t[0];
                float t1 = (half == 0) ? corner_t[1] : corner_t[2];
                float t2 = (half == 0) ? corner_t[2] : corner_t[3];
                int voff = vbase + half * nV;
                int toff = triBase + half * nT;

                for (int v = 0; v < nV; v++) {
                    float2 uv = bary_vertex_uv(v, faceSubdiv);
                    float wb = 1.0f - uv.x - uv.y;
                    float3 wp_sub = make_float3(
                        wb*v0.x + uv.x*v1.x + uv.y*v2.x,
                        wb*v0.y + uv.x*v1.y + uv.y*v2.y,
                        wb*v0.z + uv.x*v1.z + uv.y*v2.z);
                    outVerts[voff + v] = ftheta_project(wp_sub, poseData, camData);
                    float gt = wb*t0 + uv.x*t1 + uv.y*t2;
                    outVertColors[voff + v] = pack_rgba8(
                        bc_col.x + gt*(fc_col.x - bc_col.x),
                        bc_col.y + gt*(fc_col.y - bc_col.y),
                        bc_col.z + gt*(fc_col.z - bc_col.z));
                }
                float d0 = camera_depth(v0, poseData);
                float d1 = camera_depth(v1, poseData);
                float d2 = camera_depth(v2, poseData);
                for (int ti = 0; ti < nT; ti++) {
                    int3 idx = bary_triangle_indices(ti, faceSubdiv);
                    // A piece reaching behind the camera collapses to a point
                    // rather than going unwritten, its room being reserved.
                    if (!subtri_in_front(idx, faceSubdiv, d0, d1, d2, cullBehind))
                        idx = make_int3(idx.x, idx.x, idx.x);
                    outIndices[(toff + ti)*3 + 0] = voff + idx.x;
                    outIndices[(toff + ti)*3 + 1] = voff + idx.y;
                    outIndices[(toff + ti)*3 + 2] = voff + idx.z;
                }
            }
        }
    }

    // --- Lanes 6-17: wireframe edges with linear tessellation ---
    if ((renderFlags & 1u) && lane >= 6 && lane < 18) {
        int edgeIdx = lane - 6;

        int f0 = EDGE_FACES_D[edgeIdx][0], f1 = EDGE_FACES_D[edgeIdx][1];
        bool anyVisible = cameraInside;
        for (int fi = 0; fi < 2; fi++) {
            int f = (fi == 0) ? f0 : f1;
            float3 n = quat_rotate_d(qr, FACE_NORMALS_D[f]);
            float3 cl = make_float3(FACE_NORMALS_D[f].x*0.5f*sc.x, FACE_NORMALS_D[f].y*0.5f*sc.y, FACE_NORMALS_D[f].z*0.5f*sc.z);
            float3 cWld = quat_rotate_d(qr, cl);
            cWld.x += tr.x; cWld.y += tr.y; cWld.z += tr.z;
            float3 tc = make_float3(cw.x-cWld.x, cw.y-cWld.y, cw.z-cWld.z);
            if (n.x*tc.x + n.y*tc.y + n.z*tc.z > 0.0f) { anyVisible = true; break; }
        }
        if (!anyVisible) return;

        int vi0 = CUBE_EDGES_D[edgeIdx][0], vi1 = CUBE_EDGES_D[edgeIdx][1];
        float3 lv0 = CUBE_VERTS_D[vi0], lv1 = CUBE_VERTS_D[vi1];
        float3 sv0 = make_float3(lv0.x*sc.x, lv0.y*sc.y, lv0.z*sc.z);
        float3 sv1 = make_float3(lv1.x*sc.x, lv1.y*sc.y, lv1.z*sc.z);
        float3 wv0 = quat_rotate_d(qr, sv0); wv0.x += tr.x; wv0.y += tr.y; wv0.z += tr.z;
        float3 wv1 = quat_rotate_d(qr, sv1); wv1.x += tr.x; wv1.y += tr.y; wv1.z += tr.z;

        int numSegs = 1 << subdiv;
        int vbase = atomicAdd(atomicVerts, numSegs * 4);
        int triBase = atomicAdd(atomicTris, numSegs * 2);
        if (!geometry_reservation_fits(vbase, numSegs * 4, triBase, numSegs * 2,
                                       atomicVerts, atomicTris))
            return;
        float img_w = camData[2], img_h = camData[3];
        float edgeWidth = wireframeWidth;
        int edgeNdc = widthInNdc;
        float edgeScaling = depthScaling;
        uint32_t ec = pack_rgba8(0.784f, 0.784f, 0.784f);

        for (int seg = 0; seg < numSegs; seg++) {
            float ta = (float)seg / (float)numSegs;
            float tb = (float)(seg + 1) / (float)numSegs;
            float3 pa = make_float3(wv0.x + ta*(wv1.x-wv0.x), wv0.y + ta*(wv1.y-wv0.y), wv0.z + ta*(wv1.z-wv0.z));
            float3 pb = make_float3(wv0.x + tb*(wv1.x-wv0.x), wv0.y + tb*(wv1.y-wv0.y), wv0.z + tb*(wv1.z-wv0.z));
            clip_to_near(pa, pb, poseData, cullBehind);
            float4 ca = ftheta_project(pa, poseData, camData);
            float4 cb = ftheta_project(pb, poseData, camData);

            float wa = fmaxf(fabsf(ca.w), 0.001f), wb = fmaxf(fabsf(cb.w), 0.001f);
            float ox, oy;
            get_edge_offset(ca, cb, wa, wb, img_w, img_h, edgeWidth, edgeNdc,
                            edgeScaling, ox, oy);
            float zba = -0.001f*ca.w, zbb = -0.001f*cb.w;

            int sv = vbase + seg * 4;
            int st = triBase + seg * 2;
            outVerts[sv+0] = make_float4(ca.x-ox*ca.w, ca.y-oy*ca.w, ca.z+zba, ca.w);
            outVerts[sv+1] = make_float4(ca.x+ox*ca.w, ca.y+oy*ca.w, ca.z+zba, ca.w);
            outVerts[sv+2] = make_float4(cb.x+ox*cb.w, cb.y+oy*cb.w, cb.z+zbb, cb.w);
            outVerts[sv+3] = make_float4(cb.x-ox*cb.w, cb.y-oy*cb.w, cb.z+zbb, cb.w);
            outIndices[st*3+0]=sv; outIndices[st*3+1]=sv+1; outIndices[st*3+2]=sv+2;
            outIndices[(st+1)*3+0]=sv; outIndices[(st+1)*3+1]=sv+2; outIndices[(st+1)*3+2]=sv+3;
            for (int i = 0; i < 4; i++) outVertColors[sv+i] = ec;
        }
    }
}

//------------------------------------------------------------------------
// Dot primitives: one thread per dot, screen-space hexagon (matches GL octagon).
// dotPositions: [numDots, 4] float (x,y,z,pad) - same layout as Vertex
// dotRadius: half-width in pixels (e.g. 3.5 for 7px line width)

__global__ void dotGeometryKernel(
    const Vertex* __restrict__ dotPositions,
    int numDots,
    float dotRadius,
    uint32_t dotColor,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    int di = blockIdx.x * blockDim.x + threadIdx.x;
    if (di >= numDots) return;

    float3 wp = make_float3(dotPositions[di].position[0],
                            dotPositions[di].position[1],
                            dotPositions[di].position[2]);
    float4 clip = ftheta_project(wp, poseData, camData);
    float w = fmaxf(fabsf(clip.w), 1e-6f);

    // Depth-based radius scaling (matches GL get_depth_scale)
    float z_ndc = clip.z / w;
    float depth_scale = fminf(fmaxf((1.0f - z_ndc) * 0.5f, 0.0f), 1.0f);
    float r = dotRadius * depth_scale;
    if (r < 0.5f) return;

    float img_w = camData[2], img_h = camData[3];

    // 7 vertices: center + 6 ring, 6 triangles
    int vbase = atomicAdd(atomicVerts, 7);
    int triBase = atomicAdd(atomicTris, 6);
    if (!geometry_reservation_fits(vbase, 7, triBase, 6, atomicVerts, atomicTris))
        return;

    outVerts[vbase] = clip;
    outVertColors[vbase] = dotColor;

    for (int i = 0; i < 6; i++) {
        float angle = (float)i * 1.0471975f;  // 2*PI/6
        float ox = cosf(angle) * r * 2.0f / img_w * clip.w;
        float oy = -sinf(angle) * r * 2.0f / img_h * clip.w;
        outVerts[vbase + 1 + i] = make_float4(clip.x + ox, clip.y + oy, clip.z, clip.w);
        outVertColors[vbase + 1 + i] = dotColor;
    }

    for (int i = 0; i < 6; i++) {
        int next = (i + 1) % 6;
        outIndices[(triBase + i) * 3 + 0] = vbase;
        outIndices[(triBase + i) * 3 + 1] = vbase + 1 + i;
        outIndices[(triBase + i) * 3 + 2] = vbase + 1 + next;
    }
}

//------------------------------------------------------------------------
// Polyline: one block per polyline, single-threaded body generation.
// With adaptive tessellation: segments are subdivided based on f-theta distortion.

__global__ void polylineGeometryKernel(
    const PolylineHeader* __restrict__ headers, int numPolylines,
    const Vertex* __restrict__ vertices,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    float tessThreshold, int maxTessLevel, int widthInNdc, int cullBehind,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    int plIdx = blockIdx.x;
    if (plIdx >= numPolylines) return;
    if (threadIdx.x != 0) return;

    const PolylineHeader& pl = headers[plIdx];
    int numPts = (int)pl.vertex_count;
    if (numPts < 2) return;

    float img_w = camData[2];
    float img_h = camData[3];
    float half_width = pl.width * 0.5f;
    uint32_t packedColor = pack_rgba8(pl.color[0], pl.color[1], pl.color[2]);

    const int MAX_PTS = 256;
    int safePts = numPts < MAX_PTS ? numPts : MAX_PTS;
    int numSegs = safePts - 1;

    // Where the line first comes in front of the camera, which is the point
    // the ones behind are pulled back towards. A line with no such point has
    // nothing to draw and every point of it would land somewhere arbitrary,
    // so it goes before any of the buffer is spoken for.
    float3 anchor = make_float3(0.0f, 0.0f, 0.0f);
    if (cullBehind) {
        bool infront = false;
        for (int i = 0; i < safePts && !infront; i++) {
            const Vertex& vi = vertices[pl.vertex_start + i];
            float3 wi = make_float3(vi.position[0], vi.position[1], vi.position[2]);
            if (camera_depth(wi, poseData) > NEAR_DEPTH) {
                anchor = wi;
                infront = true;
            }
        }
        if (!infront) return;
    }

    // Phase 1: count total effective points (with subdivision)
    int totalEffPts = 1;
    for (int seg = 0; seg < numSegs; seg++) {
        const Vertex& va = vertices[pl.vertex_start + seg];
        const Vertex& vb = vertices[pl.vertex_start + seg + 1];
        float3 wa = make_float3(va.position[0], va.position[1], va.position[2]);
        float3 wb = make_float3(vb.position[0], vb.position[1], vb.position[2]);
        float error = estimate_edge_distortion_pixels(wa, wb, poseData, camData);
        int level = compute_subdiv_level(error, tessThreshold, maxTessLevel);
        totalEffPts += (1 << level);
    }

    // Phase 2: allocate output
    int vbase = atomicAdd(atomicVerts, totalEffPts * 2);
    int triBase = atomicAdd(atomicTris, (totalEffPts - 1) * 2);
    if (!geometry_reservation_fits(vbase, totalEffPts * 2, triBase,
                                   (totalEffPts - 1) * 2, atomicVerts, atomicTris))
        return;

    // Phase 3: generate effective points and geometry
    int ept = 0;
    float prev_sx = 0.0f, prev_sy = 0.0f;

    for (int seg = 0; seg < numSegs; seg++) {
        const Vertex& va = vertices[pl.vertex_start + seg];
        const Vertex& vb = vertices[pl.vertex_start + seg + 1];
        float3 wp_a = make_float3(va.position[0], va.position[1], va.position[2]);
        float3 wp_b = make_float3(vb.position[0], vb.position[1], vb.position[2]);
        float error = estimate_edge_distortion_pixels(wp_a, wp_b, poseData, camData);
        int level = compute_subdiv_level(error, tessThreshold, maxTessLevel);
        int N = 1 << level;

        int sub_start = (seg == 0) ? 0 : 1;
        for (int j = sub_start; j <= N; j++) {
            float t = (float)j / (float)N;
            float3 wp;
            if (j == 0) wp = wp_a;
            else if (j == N) wp = wp_b;
            else wp = make_float3(
                wp_a.x + t * (wp_b.x - wp_a.x),
                wp_a.y + t * (wp_b.y - wp_a.y),
                wp_a.z + t * (wp_b.z - wp_a.z));

            if (cullBehind) {
                if (camera_depth(wp, poseData) > NEAR_DEPTH) anchor = wp;
                else wp = pulled_to_near(anchor, wp, poseData);
            }

            float4 clip = ftheta_project(wp, poseData, camData);
            float w = fmaxf(fabsf(clip.w), 0.001f);
            float curr_sx = (clip.x / w * 0.5f + 0.5f) * img_w;
            float curr_sy = (0.5f - clip.y / w * 0.5f) * img_h;

            float z_ndc = clip.z / w;
            float depth_scale = fminf(fmaxf((1.0f - z_ndc) * 0.5f, 0.0f), 1.0f);
            float scaled_hw = half_width * depth_scale;

            float dx, dy;
            float miter_scale = 1.0f;

            if (ept == 0) {
                // First effective point: forward to next effective point
                float3 wp_next;
                if (N > 1) {
                    float tn = 1.0f / (float)N;
                    wp_next = make_float3(
                        wp_a.x + tn * (wp_b.x - wp_a.x),
                        wp_a.y + tn * (wp_b.y - wp_a.y),
                        wp_a.z + tn * (wp_b.z - wp_a.z));
                } else {
                    wp_next = wp_b;
                }
                float4 cn = ftheta_project(wp_next, poseData, camData);
                float wn = fmaxf(fabsf(cn.w), 0.001f);
                dx = (cn.x / wn * 0.5f + 0.5f) * img_w - curr_sx;
                dy = (0.5f - cn.y / wn * 0.5f) * img_h - curr_sy;
            } else if (ept == totalEffPts - 1) {
                // Last effective point: backward
                dx = curr_sx - prev_sx;
                dy = curr_sy - prev_sy;
            } else if (j == N && seg < numSegs - 1) {
                // Original vertex boundary: full miter
                float bx = curr_sx - prev_sx;
                float by = curr_sy - prev_sy;
                const Vertex& vn = vertices[pl.vertex_start + seg + 2];
                float3 wn = make_float3(vn.position[0], vn.position[1], vn.position[2]);
                float4 cn = ftheta_project(wn, poseData, camData);
                float wn2 = fmaxf(fabsf(cn.w), 0.001f);
                float fx = (cn.x / wn2 * 0.5f + 0.5f) * img_w - curr_sx;
                float fy = (0.5f - cn.y / wn2 * 0.5f) * img_h - curr_sy;

                float bl = sqrtf(bx*bx + by*by);
                float fl = sqrtf(fx*fx + fy*fy);
                if (bl > 0.001f) { bx /= bl; by /= bl; } else { bx = 1.0f; by = 0.0f; }
                if (fl > 0.001f) { fx /= fl; fy /= fl; } else { fx = 1.0f; fy = 0.0f; }

                float perp_next_x = -fy, perp_next_y = fx;
                float mx = (-by + perp_next_x), my = (bx + perp_next_y);
                float ml = sqrtf(mx*mx + my*my);
                if (ml > 0.001f) {
                    mx /= ml; my /= ml;
                    float cos_half = mx * perp_next_x + my * perp_next_y;
                    miter_scale = (cos_half > 0.5f) ? (1.0f / cos_half) : 2.0f;
                    dx = my; dy = -mx;
                } else {
                    dx = bx + fx; dy = by + fy;
                }
            } else {
                // Interior sub-point: smooth curve, backward direction suffices
                dx = curr_sx - prev_sx;
                dy = curr_sy - prev_sy;
            }

            float dl = sqrtf(dx*dx + dy*dy);
            if (dl > 0.001f) { dx /= dl; dy /= dl; } else { dx = 1.0f; dy = 0.0f; }

            float ox, oy;
            get_line_offset(dx, dy, scaled_hw, miter_scale,
                            img_w, img_h, widthInNdc, clip.w, ox, oy);

            outVerts[vbase + ept*2]     = make_float4(clip.x - ox, clip.y - oy, clip.z, clip.w);
            outVerts[vbase + ept*2 + 1] = make_float4(clip.x + ox, clip.y + oy, clip.z, clip.w);
            outVertColors[vbase + ept*2]     = packedColor;
            outVertColors[vbase + ept*2 + 1] = packedColor;

            prev_sx = curr_sx;
            prev_sy = curr_sy;
            ept++;
        }
    }

    for (int i = 0; i < totalEffPts - 1; i++) {
        int b = vbase + i * 2;
        int tOut = triBase + i * 2;
        outIndices[tOut*3 + 0] = b;     outIndices[tOut*3 + 1] = b + 1; outIndices[tOut*3 + 2] = b + 2;
        outIndices[(tOut+1)*3 + 0] = b + 1; outIndices[(tOut+1)*3 + 1] = b + 3; outIndices[(tOut+1)*3 + 2] = b + 2;
    }
}

//------------------------------------------------------------------------
// Timestamped polyline pool kernel.
// One block per (pool, varray_offset). Thread 0 does timestamp search;
// single-threaded geometry generation (same as polylineGeometryKernel).
//------------------------------------------------------------------------

__global__ void polylinePoolKernel(
    const TsPolylinePoolHeader* __restrict__ poolHeaders, int numPools,
    const int64_t* __restrict__ allTimestamps,
    const int32_t* __restrict__ allInt32,
    const Vertex* __restrict__ allVertices,
    const float* __restrict__ allFloats,
    int64_t queryTs,
    int maxVarraysPerPool,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    CudaRenderParams params,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    if (threadIdx.x != 0) return;

    int workId = blockIdx.x;
    int varrayOffset = workId % maxVarraysPerPool;
    int poolId = workId / maxVarraysPerPool;
    if (poolId >= numPools) return;

    const TsPolylinePoolHeader& pool = poolHeaders[poolId];

    int tsIdx = binary_search_timestamps(allTimestamps, pool.timestamps_offset, pool.num_timestamps, queryTs);
    if (tsIdx < 0) return;

    uint32_t varrayStart = (tsIdx > 0) ? (uint32_t)allInt32[pool.ts_varrays_ps_offset + tsIdx - 1] : 0u;
    uint32_t varrayEnd   = (uint32_t)allInt32[pool.ts_varrays_ps_offset + tsIdx];
    uint32_t numVarrays  = varrayEnd - varrayStart;
    if ((uint32_t)varrayOffset >= numVarrays) return;

    uint32_t actualIdx = varrayStart + (uint32_t)varrayOffset;

    // Spatial culling via AABB
    if (params.cullRadiusScale > 0.0f) {
        float cull_r = camData[13] * params.cullRadiusScale;  // depth_max * scale
        float3 cw = cube_cam_world(poseData);
        uint32_t aabbBase = pool.aabb_offset + actualIdx * 6;
        float minx = allFloats[aabbBase], miny = allFloats[aabbBase+1], minz = allFloats[aabbBase+2];
        float maxx = allFloats[aabbBase+3], maxy = allFloats[aabbBase+4], maxz = allFloats[aabbBase+5];
        if (maxx < cw.x - cull_r || minx > cw.x + cull_r ||
            maxy < cw.y - cull_r || miny > cw.y + cull_r ||
            maxz < cw.z - cull_r || minz > cw.z + cull_r)
            return;
    }

    uint32_t vStart = (actualIdx > 0) ? (uint32_t)allInt32[pool.varrays_ps_offset + actualIdx - 1] : 0u;
    uint32_t vEnd   = (uint32_t)allInt32[pool.varrays_ps_offset + actualIdx];
    int numPts = (int)(vEnd - vStart);
    if (numPts < 1) return;

    uint32_t packedColor = get_prim_color_packed(pool.prim_type_id, params);

    // Dot primitives: render each vertex as a hexagon
    if (is_dot_primitive(pool.prim_type_id)) {
        float dotRadius = get_prim_width(pool.prim_type_id, params) * 0.5f;
        float img_w = camData[2], img_h = camData[3];
        for (int di = 0; di < numPts; di++) {
            const Vertex& vt = allVertices[pool.vertices_offset + vStart + di];
            float3 wp = make_float3(vt.position[0], vt.position[1], vt.position[2]);
            // A dot behind the camera has nowhere on the image to be, and a
            // hexagon drawn where the projection puts it instead is the one
            // thing on screen that was never on the road.
            if (params.cullBehindCamera && camera_depth(wp, poseData) <= NEAR_DEPTH)
                continue;
            float4 clip = ftheta_project(wp, poseData, camData);
            float w = fmaxf(fabsf(clip.w), 1e-6f);
            float z_ndc = clip.z / w;
            float depth_scale = (params.depthScaling > 0.5f)
                ? fminf(fmaxf((1.0f - z_ndc) * 0.5f, 0.0f), 1.0f) : 1.0f;
            float r = dotRadius * depth_scale;
            if (r < 0.5f) continue;
            int vbase = atomicAdd(atomicVerts, 7);
            int triBase = atomicAdd(atomicTris, 6);
            // Carry on rather than give up on the rest: the retry sizes its
            // buffers from these counters, and a dot that never reserves is a
            // dot the second attempt has no room for either.
            if (!geometry_reservation_fits(vbase, 7, triBase, 6, atomicVerts, atomicTris))
                continue;
            outVerts[vbase] = clip;
            outVertColors[vbase] = packedColor;
            for (int i = 0; i < 6; i++) {
                float angle = (float)i * 1.0471975f;
                // Round in whichever convention lines are drawn in, a dot
                // being a cap with no line to it.
                float ox, oy;
                get_cap_offset(1.0f, 0.0f, r, sinf(angle), cosf(angle),
                               img_w, img_h, params.widthInNdc, clip.w, ox, oy);
                outVerts[vbase + 1 + i] = make_float4(clip.x + ox, clip.y + oy, clip.z, clip.w);
                outVertColors[vbase + 1 + i] = packedColor;
            }
            for (int i = 0; i < 6; i++) {
                int next = (i + 1) % 6;
                outIndices[(triBase + i) * 3 + 0] = vbase;
                outIndices[(triBase + i) * 3 + 1] = vbase + 1 + i;
                outIndices[(triBase + i) * 3 + 2] = vbase + 1 + next;
            }
        }
        return;
    }

    // Regular polyline
    if (numPts < 2) return;

    float img_w = camData[2], img_h = camData[3];
    float half_width = get_prim_width(pool.prim_type_id, params) * 0.5f;
    int maxTessLvl = (params.maxTessPolyline >= 0) ? params.maxTessPolyline : 4;

    const int MAX_PTS = 256;
    int safePts = numPts < MAX_PTS ? numPts : MAX_PTS;
    int numSegs = safePts - 1;

    // The point behind the camera are pulled back towards; a line that never
    // comes in front of it is dropped, before any of the buffer is spoken for.
    float3 anchor = make_float3(0.0f, 0.0f, 0.0f);
    if (params.cullBehindCamera) {
        bool infront = false;
        for (int i = 0; i < safePts && !infront; i++) {
            const Vertex& vi = allVertices[pool.vertices_offset + vStart + i];
            float3 wi = make_float3(vi.position[0], vi.position[1], vi.position[2]);
            if (camera_depth(wi, poseData) > NEAR_DEPTH) {
                anchor = wi;
                infront = true;
            }
        }
        if (!infront) return;
    }

    int totalEffPts = 1;
    for (int seg = 0; seg < numSegs; seg++) {
        const Vertex& va = allVertices[pool.vertices_offset + vStart + seg];
        const Vertex& vb = allVertices[pool.vertices_offset + vStart + seg + 1];
        float3 wa = make_float3(va.position[0], va.position[1], va.position[2]);
        float3 wb = make_float3(vb.position[0], vb.position[1], vb.position[2]);
        float error = estimate_edge_distortion_pixels(wa, wb, poseData, camData);
        int level = compute_subdiv_level(error, params.tessellationThreshold, maxTessLvl);
        totalEffPts += (1 << level);
    }

    const int CAP_SEGS = 6;
    int vbase = atomicAdd(atomicVerts, totalEffPts * 2 + CAP_SEGS * 2);
    int triBase = atomicAdd(atomicTris, (totalEffPts - 1) * 2 + CAP_SEGS * 2);
    if (!geometry_reservation_fits(vbase, totalEffPts * 2 + CAP_SEGS * 2,
                                   triBase, (totalEffPts - 1) * 2 + CAP_SEGS * 2,
                                   atomicVerts, atomicTris))
        return;

    int ept = 0;
    float prev_sx = 0.0f, prev_sy = 0.0f;
    float4 capClip[2];
    float capDx[2], capDy[2], capHW[2];

    for (int seg = 0; seg < numSegs; seg++) {
        const Vertex& va = allVertices[pool.vertices_offset + vStart + seg];
        const Vertex& vb = allVertices[pool.vertices_offset + vStart + seg + 1];
        float3 wp_a = make_float3(va.position[0], va.position[1], va.position[2]);
        float3 wp_b = make_float3(vb.position[0], vb.position[1], vb.position[2]);
        float error = estimate_edge_distortion_pixels(wp_a, wp_b, poseData, camData);
        int level = compute_subdiv_level(error, params.tessellationThreshold, maxTessLvl);
        int N = 1 << level;

        int sub_start = (seg == 0) ? 0 : 1;
        for (int j = sub_start; j <= N; j++) {
            float t = (float)j / (float)N;
            float3 wp;
            if (j == 0) wp = wp_a;
            else if (j == N) wp = wp_b;
            else wp = make_float3(
                wp_a.x + t * (wp_b.x - wp_a.x),
                wp_a.y + t * (wp_b.y - wp_a.y),
                wp_a.z + t * (wp_b.z - wp_a.z));

            if (params.cullBehindCamera) {
                if (camera_depth(wp, poseData) > NEAR_DEPTH) anchor = wp;
                else wp = pulled_to_near(anchor, wp, poseData);
            }

            float4 clip = ftheta_project(wp, poseData, camData);
            float w = fmaxf(fabsf(clip.w), 0.001f);
            float curr_sx = (clip.x / w * 0.5f + 0.5f) * img_w;
            float curr_sy = (0.5f - clip.y / w * 0.5f) * img_h;
            float z_ndc = clip.z / w;
            float depth_scale = (params.depthScaling > 0.5f)
                ? fminf(fmaxf((1.0f - z_ndc) * 0.5f, 0.0f), 1.0f) : 1.0f;
            float scaled_hw = half_width * depth_scale;

            float dx, dy;
            float miter_scale = 1.0f;

            if (ept == 0) {
                float3 wp_next;
                if (N > 1) {
                    float tn = 1.0f / (float)N;
                    wp_next = make_float3(
                        wp_a.x + tn * (wp_b.x - wp_a.x),
                        wp_a.y + tn * (wp_b.y - wp_a.y),
                        wp_a.z + tn * (wp_b.z - wp_a.z));
                } else {
                    wp_next = wp_b;
                }
                float4 cn = ftheta_project(wp_next, poseData, camData);
                float wn = fmaxf(fabsf(cn.w), 0.001f);
                dx = (cn.x / wn * 0.5f + 0.5f) * img_w - curr_sx;
                dy = (0.5f - cn.y / wn * 0.5f) * img_h - curr_sy;
            } else if (ept == totalEffPts - 1) {
                dx = curr_sx - prev_sx;
                dy = curr_sy - prev_sy;
            } else if (j == N && seg < numSegs - 1) {
                float bx = curr_sx - prev_sx;
                float by = curr_sy - prev_sy;
                const Vertex& vn = allVertices[pool.vertices_offset + vStart + seg + 2];
                float3 wn = make_float3(vn.position[0], vn.position[1], vn.position[2]);
                float4 cn = ftheta_project(wn, poseData, camData);
                float wn2 = fmaxf(fabsf(cn.w), 0.001f);
                float fx = (cn.x / wn2 * 0.5f + 0.5f) * img_w - curr_sx;
                float fy = (0.5f - cn.y / wn2 * 0.5f) * img_h - curr_sy;
                float bl = sqrtf(bx*bx + by*by);
                float fl = sqrtf(fx*fx + fy*fy);
                if (bl > 0.001f) { bx /= bl; by /= bl; } else { bx = 1.0f; by = 0.0f; }
                if (fl > 0.001f) { fx /= fl; fy /= fl; } else { fx = 1.0f; fy = 0.0f; }
                float perp_next_x = -fy, perp_next_y = fx;
                float mx = (-by + perp_next_x), my = (bx + perp_next_y);
                float ml = sqrtf(mx*mx + my*my);
                if (ml > 0.001f) {
                    mx /= ml; my /= ml;
                    float cos_half = mx * perp_next_x + my * perp_next_y;
                    miter_scale = (cos_half > 0.5f) ? (1.0f / cos_half) : 2.0f;
                    dx = my; dy = -mx;
                } else {
                    dx = bx + fx; dy = by + fy;
                }
            } else {
                dx = curr_sx - prev_sx;
                dy = curr_sy - prev_sy;
            }

            float dl = sqrtf(dx*dx + dy*dy);
            if (dl > 0.001f) { dx /= dl; dy /= dl; } else { dx = 1.0f; dy = 0.0f; }
            // Only where the line ends and which way it was going: the cap
            // builds its own perpendicular, in the convention it is drawn in.
            if (ept == 0) {
                capClip[0]=clip; capDx[0]=dx; capDy[0]=dy; capHW[0]=scaled_hw;
            }
            if (ept == totalEffPts - 1) {
                capClip[1]=clip; capDx[1]=dx; capDy[1]=dy; capHW[1]=scaled_hw;
            }
            float ox, oy;
            get_line_offset(dx, dy, scaled_hw, miter_scale,
                            img_w, img_h, params.widthInNdc, clip.w, ox, oy);
            outVerts[vbase + ept*2]     = make_float4(clip.x - ox, clip.y - oy, clip.z, clip.w);
            outVerts[vbase + ept*2 + 1] = make_float4(clip.x + ox, clip.y + oy, clip.z, clip.w);
            outVertColors[vbase + ept*2]     = packedColor;
            outVertColors[vbase + ept*2 + 1] = packedColor;
            prev_sx = curr_sx;
            prev_sy = curr_sy;
            ept++;
        }
    }

    // Quad strip indices
    for (int i = 0; i < totalEffPts - 1; i++) {
        int b = vbase + i * 2;
        int tOut = triBase + i * 2;
        outIndices[tOut*3 + 0] = b;     outIndices[tOut*3 + 1] = b + 1; outIndices[tOut*3 + 2] = b + 2;
        outIndices[(tOut+1)*3 + 0] = b + 1; outIndices[(tOut+1)*3 + 1] = b + 3; outIndices[(tOut+1)*3 + 2] = b + 2;
    }

    // Round caps: semicircular triangle fans at both endpoints
    int capV = vbase + totalEffPts * 2;
    int capT = triBase + (totalEffPts - 1) * 2;
    for (int cap = 0; cap < 2; cap++) {
        float4 c = capClip[cap];
        float hw = capHW[cap];
        float fwd_sign = (cap == 0) ? -1.0f : 1.0f;

        // Existing side vertices for this endpoint
        int eptIdx = (cap == 0) ? 0 : (totalEffPts - 1);
        int existPlus  = vbase + eptIdx * 2 + 1;
        int existMinus = vbase + eptIdx * 2;

        // Center vertex
        outVerts[capV] = c;
        outVertColors[capV] = packedColor;

        // Intermediate arc vertices (theta from pi/N to (N-1)*pi/N)
        for (int i = 1; i < CAP_SEGS; i++) {
            float theta = (float)i * 3.14159265f / (float)CAP_SEGS;
            float aox, aoy;
            get_cap_offset(capDx[cap], capDy[cap], hw,
                           cosf(theta), sinf(theta) * fwd_sign,
                           img_w, img_h, params.widthInNdc, c.w, aox, aoy);
            outVerts[capV + i] = make_float4(c.x + aox, c.y + aoy, c.z, c.w);
            outVertColors[capV + i] = packedColor;
        }

        // Fan triangles: plus -> arc[1..N-1] -> minus
        outIndices[capT * 3 + 0] = capV;
        outIndices[capT * 3 + 1] = existPlus;
        outIndices[capT * 3 + 2] = capV + 1;
        for (int i = 1; i < CAP_SEGS - 1; i++) {
            outIndices[(capT + i) * 3 + 0] = capV;
            outIndices[(capT + i) * 3 + 1] = capV + i;
            outIndices[(capT + i) * 3 + 2] = capV + i + 1;
        }
        outIndices[(capT + CAP_SEGS - 1) * 3 + 0] = capV;
        outIndices[(capT + CAP_SEGS - 1) * 3 + 1] = capV + CAP_SEGS - 1;
        outIndices[(capT + CAP_SEGS - 1) * 3 + 2] = existMinus;

        capV += CAP_SEGS;
        capT += CAP_SEGS;
    }
}

//------------------------------------------------------------------------
// Timestamped polygon pool kernel.
// One block per (pool, varray_offset), 64 threads per block.
// Thread 0 does timestamp search; all threads generate triangles.
//------------------------------------------------------------------------

__global__ void polygonPoolKernel(
    const TsPolygonPoolHeader* __restrict__ poolHeaders, int numPools,
    const int64_t* __restrict__ allTimestamps,
    const int32_t* __restrict__ allInt32,
    const Vertex* __restrict__ allVertices,
    const Triangle* __restrict__ allTriangles,
    const float* __restrict__ allFloats,
    int64_t queryTs,
    int maxVarraysPerPool,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    CudaRenderParams params,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    __shared__ int s_triStart, s_triEnd, s_vStart;
    __shared__ uint32_t s_color;

    int workId = blockIdx.x;
    int varrayOffset = workId % maxVarraysPerPool;
    int poolId = workId / maxVarraysPerPool;
    if (poolId >= numPools) return;

    const TsPolygonPoolHeader& pool = poolHeaders[poolId];
    if (pool.prim_type_id == PRIM_BEV_ROAD_SURFACE &&
        params.cameraTypeId != CAMERA_TYPE_BEV) return;

    if (threadIdx.x == 0) {
        s_triStart = -1;  // sentinel: "no work"

        int tsIdx = binary_search_timestamps(allTimestamps, pool.timestamps_offset, pool.num_timestamps, queryTs);
        if (tsIdx >= 0) {
            uint32_t varrayStart = (tsIdx > 0) ? (uint32_t)allInt32[pool.ts_varrays_ps_offset + tsIdx - 1] : 0u;
            uint32_t varrayEnd   = (uint32_t)allInt32[pool.ts_varrays_ps_offset + tsIdx];
            uint32_t numVarrays  = varrayEnd - varrayStart;

            if ((uint32_t)varrayOffset < numVarrays) {
                uint32_t actualIdx = varrayStart + (uint32_t)varrayOffset;

                // Spatial culling via AABB
                bool culled = false;
                if (params.cullRadiusScale > 0.0f) {
                    float cull_r = camData[13] * params.cullRadiusScale;
                    float3 cw = cube_cam_world(poseData);
                    uint32_t aabbBase = pool.aabb_offset + actualIdx * 6;
                    float minx = allFloats[aabbBase], miny = allFloats[aabbBase+1], minz = allFloats[aabbBase+2];
                    float maxx = allFloats[aabbBase+3], maxy = allFloats[aabbBase+4], maxz = allFloats[aabbBase+5];
                    if (maxx < cw.x - cull_r || minx > cw.x + cull_r ||
                        maxy < cw.y - cull_r || miny > cw.y + cull_r ||
                        maxz < cw.z - cull_r || minz > cw.z + cull_r)
                        culled = true;
                }

                if (!culled) {
                    uint32_t triS = (actualIdx > 0) ? (uint32_t)allInt32[pool.tri_ps_offset + actualIdx - 1] : 0u;
                    uint32_t triE = (uint32_t)allInt32[pool.tri_ps_offset + actualIdx];

                    if (triE > triS) {
                        uint32_t vS = (actualIdx > 0) ? (uint32_t)allInt32[pool.varrays_ps_offset + actualIdx - 1] : 0u;
                        s_triStart = (int)triS;
                        s_triEnd   = (int)triE;
                        s_vStart   = (int)vS;
                        s_color    = get_prim_color_packed(pool.prim_type_id, params);
                    }
                }
            }
        }
    }
    __syncthreads();

    int triStart = s_triStart;
    if (triStart < 0) return;
    int triEnd = s_triEnd;
    int vStartLocal = s_vStart;
    uint32_t color = s_color;
    int totalTris = triEnd - triStart;
    int maxTessLvl = (params.maxTessPolygon >= 0) ? params.maxTessPolygon : 3;

    for (int t = (int)threadIdx.x; t < totalTris; t += (int)blockDim.x) {
        const Triangle& tri = allTriangles[pool.triangles_offset + triStart + t];
        float3 wp[3];
        for (int vi = 0; vi < 3; vi++) {
            uint32_t li = tri.indices[vi];
            const Vertex& vtx = allVertices[pool.vertices_offset + vStartLocal + li];
            wp[vi] = make_float3(vtx.position[0], vtx.position[1], vtx.position[2]);
        }

        float3 kept[2][3];
        int faces = near_clipped_triangle(wp, poseData, params.cullBehindCamera, kept);

        // A piece that finds no room is passed over rather than ending the
        // thread: the retry sizes its buffers from these counters, so work
        // that never reserves is work it leaves no room for either.
        for (int face = 0; face < faces; face++) {
            const float3* fp = kept[face];

            int level = 0;
            if (params.tessellationThreshold > 0.0f) {
                float e01 = estimate_edge_distortion_pixels(fp[0], fp[1], poseData, camData);
                float e12 = estimate_edge_distortion_pixels(fp[1], fp[2], poseData, camData);
                float e20 = estimate_edge_distortion_pixels(fp[2], fp[0], poseData, camData);
                float emax = fmaxf(e01, fmaxf(e12, e20));
                level = compute_subdiv_level_polygon(emax, params.tessellationThreshold, maxTessLvl);
            }

            if (level == 0) {
                int vbase = atomicAdd(atomicVerts, 3);
                int triOut = atomicAdd(atomicTris, 1);
                if (!geometry_reservation_fits(vbase, 3, triOut, 1, atomicVerts, atomicTris))
                    continue;
                for (int vi = 0; vi < 3; vi++) {
                    outVerts[vbase + vi] = ftheta_project(fp[vi], poseData, camData);
                    outVertColors[vbase + vi] = color;
                }
                outIndices[triOut * 3 + 0] = vbase;
                outIndices[triOut * 3 + 1] = vbase + 1;
                outIndices[triOut * 3 + 2] = vbase + 2;
            } else {
                int nV = bary_vertex_count(level);
                int nT = bary_triangle_count(level);
                int vbase = atomicAdd(atomicVerts, nV);
                int triBase = atomicAdd(atomicTris, nT);
                if (!geometry_reservation_fits(vbase, nV, triBase, nT, atomicVerts, atomicTris))
                    continue;
                for (int v = 0; v < nV; v++) {
                    float2 uv = bary_vertex_uv(v, level);
                    float wb = 1.0f - uv.x - uv.y;
                    float3 wp_sub = make_float3(
                        wb * fp[0].x + uv.x * fp[1].x + uv.y * fp[2].x,
                        wb * fp[0].y + uv.x * fp[1].y + uv.y * fp[2].y,
                        wb * fp[0].z + uv.x * fp[1].z + uv.y * fp[2].z);
                    outVerts[vbase + v] = ftheta_project(wp_sub, poseData, camData);
                    outVertColors[vbase + v] = color;
                }
                for (int ti = 0; ti < nT; ti++) {
                    int3 idx = bary_triangle_indices(ti, level);
                    outIndices[(triBase + ti) * 3 + 0] = vbase + idx.x;
                    outIndices[(triBase + ti) * 3 + 1] = vbase + idx.y;
                    outIndices[(triBase + ti) * 3 + 2] = vbase + idx.z;
                }
            }
        }
    }
}

//------------------------------------------------------------------------
// Timestamped cube pool kernel (flat-buffer variant).
// Same algorithm as cubePoolFusedKernel but reads pool header + flat buffers
// instead of raw CubePoolParams pointers.
//------------------------------------------------------------------------

__global__ void cubePoolFlatKernel(
    const TsCubePoolHeader* __restrict__ poolHeaders, int numPools,
    const int64_t* __restrict__ allTimestamps,
    const int32_t* __restrict__ allInt32,
    const float* __restrict__ allFloats,
    int64_t queryTs, int maxExtrapUs,
    const float* __restrict__ camData,
    const float* __restrict__ poseData,
    CudaRenderParams params,
    float4* __restrict__ outVerts,
    int* __restrict__ outIndices,
    uint32_t* __restrict__ outVertColors,
    int* __restrict__ atomicVerts,
    int* __restrict__ atomicTris)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    int lane = threadIdx.x & 31;
    int warpInBlock = threadIdx.x / 32;

    int poolId = blockIdx.y;
    if (poolId >= numPools) return;
    const TsCubePoolHeader& pool = poolHeaders[poolId];
    int cubeIdx = blockIdx.x * CUBES_PER_BLOCK + warpInBlock;
    if (cubeIdx >= (int)pool.num_cubes) return;

    // BEV: skip ego_obstacle (prim_type_id == 6) when camera is not BEV
    if (pool.prim_type_id == 6 && params.cameraTypeId != 1) return;

    const float* poolTrans = allFloats + pool.translations_offset;
    const float* poolQuat  = allFloats + pool.quaternions_offset;
    const float* poolScales = allFloats + pool.scales_offset;
    const float* poolColors = allFloats + pool.colors_offset;
    const int32_t* prefixSum = allInt32 + pool.cube_ts_ps_offset;
    const int64_t* trackTs  = allTimestamps + pool.track_timestamps_offset;

    uint32_t renderFlags = pool.render_flags;

    float tr_x, tr_y, tr_z;
    float qx, qy, qz, qw;
    float sc_x, sc_y, sc_z;
    float fc_r, fc_g, fc_b;
    float bc_r, bc_g, bc_b;
    bool visible = false;
    int subdiv = 0;

    if (lane == 0) {
        cubePoolInterpolate(trackTs, prefixSum, poolTrans, poolQuat,
                            cubeIdx, queryTs, maxExtrapUs,
                            tr_x, tr_y, tr_z, qx, qy, qz, qw, visible);
        if (visible) {
            sc_x = poolScales[cubeIdx*3]; sc_y = poolScales[cubeIdx*3+1]; sc_z = poolScales[cubeIdx*3+2];
            fc_r = poolColors[cubeIdx*6]; fc_g = poolColors[cubeIdx*6+1]; fc_b = poolColors[cubeIdx*6+2];
            bc_r = poolColors[cubeIdx*6+3]; bc_g = poolColors[cubeIdx*6+4]; bc_b = poolColors[cubeIdx*6+5];

            // Spatial culling: sphere (translation + max scale) vs view box
            if (params.cullRadiusScale > 0.0f) {
                float cull_r = camData[13] * params.cullRadiusScale;
                float3 cw_cull = cube_cam_world(poseData);
                float maxSc = fmaxf(sc_x, fmaxf(sc_y, sc_z));
                float dx = tr_x - cw_cull.x, dy = tr_y - cw_cull.y, dz = tr_z - cw_cull.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist - maxSc > cull_r) visible = false;
            }
        }
        if (visible && params.tessellationThreshold > 0.0f) {
            int maxTessLvl = (params.maxTessCube >= 0) ? params.maxTessCube : 3;
            float3 tr_l = make_float3(tr_x, tr_y, tr_z);
            float4 qr_l = make_float4(qx, qy, qz, qw);
            float3 sc_l = make_float3(sc_x, sc_y, sc_z);
            float emax = 0.0f;
            for (int e = 0; e < 12; e++) {
                int i0 = CUBE_EDGES_D[e][0], i1 = CUBE_EDGES_D[e][1];
                float3 lv0 = make_float3(CUBE_VERTS_D[i0].x*sc_l.x, CUBE_VERTS_D[i0].y*sc_l.y, CUBE_VERTS_D[i0].z*sc_l.z);
                float3 lv1 = make_float3(CUBE_VERTS_D[i1].x*sc_l.x, CUBE_VERTS_D[i1].y*sc_l.y, CUBE_VERTS_D[i1].z*sc_l.z);
                float3 wv0 = quat_rotate_d(qr_l, lv0); wv0.x += tr_l.x; wv0.y += tr_l.y; wv0.z += tr_l.z;
                float3 wv1 = quat_rotate_d(qr_l, lv1); wv1.x += tr_l.x; wv1.y += tr_l.y; wv1.z += tr_l.z;
                float err = estimate_edge_distortion_pixels(wv0, wv1, poseData, camData);
                if (err > emax) emax = err;
            }
            subdiv = compute_subdiv_level_polygon(emax, params.tessellationThreshold, maxTessLvl);
        }
    }

    // Broadcast from lane 0
    visible = __shfl_sync(FULL_MASK, (int)visible, 0);
    if (!visible) return;
    tr_x = __shfl_sync(FULL_MASK, tr_x, 0); tr_y = __shfl_sync(FULL_MASK, tr_y, 0); tr_z = __shfl_sync(FULL_MASK, tr_z, 0);
    qx = __shfl_sync(FULL_MASK, qx, 0); qy = __shfl_sync(FULL_MASK, qy, 0);
    qz = __shfl_sync(FULL_MASK, qz, 0); qw = __shfl_sync(FULL_MASK, qw, 0);
    sc_x = __shfl_sync(FULL_MASK, sc_x, 0); sc_y = __shfl_sync(FULL_MASK, sc_y, 0); sc_z = __shfl_sync(FULL_MASK, sc_z, 0);
    fc_r = __shfl_sync(FULL_MASK, fc_r, 0); fc_g = __shfl_sync(FULL_MASK, fc_g, 0); fc_b = __shfl_sync(FULL_MASK, fc_b, 0);
    bc_r = __shfl_sync(FULL_MASK, bc_r, 0); bc_g = __shfl_sync(FULL_MASK, bc_g, 0); bc_b = __shfl_sync(FULL_MASK, bc_b, 0);
    subdiv = __shfl_sync(FULL_MASK, subdiv, 0);
    renderFlags = __shfl_sync(FULL_MASK, (int)renderFlags, 0);

    float4 qr = make_float4(qx, qy, qz, qw);
    float3 tr = make_float3(tr_x, tr_y, tr_z);
    float3 sc = make_float3(sc_x, sc_y, sc_z);
    if (!cube_in_front(tr, qr, sc, poseData, params.cullBehindCamera))
        return;

    float3 cw = cube_cam_world(poseData);
    bool cameraInside = point_inside_cube_d(cw, tr, qr, sc);

    // Lanes 0-5: face geometry
    if (lane < 6) {
        int faceIdx = lane;
        float3 nLocal = FACE_NORMALS_D[faceIdx];
        float3 nWorld = quat_rotate_d(qr, nLocal);
        float3 fcLocal = make_float3(
            nLocal.x * 0.5f * sc.x,
            nLocal.y * 0.5f * sc.y,
            nLocal.z * 0.5f * sc.z);
        float3 fcWorld = quat_rotate_d(qr, fcLocal);
        fcWorld.x += tr.x; fcWorld.y += tr.y; fcWorld.z += tr.z;
        float3 toCamera = make_float3(
            cw.x - fcWorld.x, cw.y - fcWorld.y, cw.z - fcWorld.z);
        if (!cameraInside &&
            nWorld.x*toCamera.x + nWorld.y*toCamera.y + nWorld.z*toCamera.z <= 0.0f)
            return;

        int i0 = FACE_VERTS_D[faceIdx][0], i1 = FACE_VERTS_D[faceIdx][1];
        int i2 = FACE_VERTS_D[faceIdx][2], i3 = FACE_VERTS_D[faceIdx][3];

        float3 corners[4];
        for (int ci = 0; ci < 4; ci++) {
            int vi = (ci == 0) ? i0 : (ci == 1) ? i1 : (ci == 2) ? i2 : i3;
            float3 lv = make_float3(CUBE_VERTS_D[vi].x * sc.x, CUBE_VERTS_D[vi].y * sc.y, CUBE_VERTS_D[vi].z * sc.z);
            float3 wv = quat_rotate_d(qr, lv);
            corners[ci] = make_float3(wv.x + tr.x, wv.y + tr.y, wv.z + tr.z);
        }
        if (!face_in_front(corners, poseData, params.cullBehindCamera))
            return;

        float corner_t[4];
        for (int ci = 0; ci < 4; ci++) {
            int vi = (ci == 0) ? i0 : (ci == 1) ? i1 : (ci == 2) ? i2 : i3;
            // CUBE_FLAG_FLAT_FACES holds a face to one colour, the front face
            // to the front colour and the other five to the back.
            corner_t[ci] = (renderFlags & 2u)
                ? ((faceIdx == 3) ? 1.0f : 0.0f)
                : (CUBE_VERTS_D[vi].x + 0.5f);
        }

        // A face across the plane is cut along its tessellation seams, so it is
        // given some even where none were asked for; only a face entirely in
        // front of the camera still takes the plain four-corner path below.
        int faceSubdiv = subdiv_for_crossing(
            corners, poseData, params.cullBehindCamera, subdiv);

        if (faceSubdiv == 0) {
            int vbase = atomicAdd(atomicVerts, 4);
            int triBase = atomicAdd(atomicTris, 2);
            if (!geometry_reservation_fits(vbase, 4, triBase, 2, atomicVerts, atomicTris))
                return;
            for (int ci = 0; ci < 4; ci++) {
                outVerts[vbase + ci] = ftheta_project(corners[ci], poseData, camData);
                float gt = corner_t[ci];
                float r = bc_r + gt * (fc_r - bc_r);
                float g = bc_g + gt * (fc_g - bc_g);
                float b = bc_b + gt * (fc_b - bc_b);
                outVertColors[vbase + ci] = pack_rgba8(r, g, b);
            }
            outIndices[triBase*3+0]=vbase; outIndices[triBase*3+1]=vbase+1; outIndices[triBase*3+2]=vbase+2;
            outIndices[(triBase+1)*3+0]=vbase; outIndices[(triBase+1)*3+1]=vbase+2; outIndices[(triBase+1)*3+2]=vbase+3;
        } else {
            for (int half = 0; half < 2; half++) {
                float3 tw[3];
                float tt[3];
                if (half == 0) {
                    tw[0] = corners[0]; tw[1] = corners[1]; tw[2] = corners[2];
                    tt[0] = corner_t[0]; tt[1] = corner_t[1]; tt[2] = corner_t[2];
                } else {
                    tw[0] = corners[0]; tw[1] = corners[2]; tw[2] = corners[3];
                    tt[0] = corner_t[0]; tt[1] = corner_t[2]; tt[2] = corner_t[3];
                }
                int nV = bary_vertex_count(faceSubdiv);
                int nT = bary_triangle_count(faceSubdiv);
                int vbase = atomicAdd(atomicVerts, nV);
                int triBase = atomicAdd(atomicTris, nT);
                if (!geometry_reservation_fits(vbase, nV, triBase, nT,
                                               atomicVerts, atomicTris))
                    return;
                for (int v = 0; v < nV; v++) {
                    float2 uv = bary_vertex_uv(v, faceSubdiv);
                    float wb = 1.0f - uv.x - uv.y;
                    float3 wp_sub = make_float3(
                        wb*tw[0].x + uv.x*tw[1].x + uv.y*tw[2].x,
                        wb*tw[0].y + uv.x*tw[1].y + uv.y*tw[2].y,
                        wb*tw[0].z + uv.x*tw[1].z + uv.y*tw[2].z);
                    outVerts[vbase + v] = ftheta_project(wp_sub, poseData, camData);
                    float gt = wb*tt[0] + uv.x*tt[1] + uv.y*tt[2];
                    float r = bc_r + gt*(fc_r-bc_r), g = bc_g + gt*(fc_g-bc_g), b = bc_b + gt*(fc_b-bc_b);
                    outVertColors[vbase + v] = pack_rgba8(r, g, b);
                }
                float d0 = camera_depth(tw[0], poseData);
                float d1 = camera_depth(tw[1], poseData);
                float d2 = camera_depth(tw[2], poseData);
                for (int ti = 0; ti < nT; ti++) {
                    int3 idx = bary_triangle_indices(ti, faceSubdiv);
                    // A piece reaching behind the camera collapses to a point
                    // rather than going unwritten, its room being reserved.
                    if (!subtri_in_front(idx, faceSubdiv, d0, d1, d2,
                                         params.cullBehindCamera))
                        idx = make_int3(idx.x, idx.x, idx.x);
                    outIndices[(triBase+ti)*3+0]=vbase+idx.x;
                    outIndices[(triBase+ti)*3+1]=vbase+idx.y;
                    outIndices[(triBase+ti)*3+2]=vbase+idx.z;
                }
            }
        }
    }

    // Lanes 6-17: wireframe edges (with backface culling, matching cubePoolFusedKernel)
    if ((renderFlags & 1u) && lane >= 6 && lane < 18) {
        int edgeIdx = lane - 6;

        int f0 = EDGE_FACES_D[edgeIdx][0], f1 = EDGE_FACES_D[edgeIdx][1];
        bool anyVisible = cameraInside;
        for (int fi = 0; fi < 2; fi++) {
            int f = (fi == 0) ? f0 : f1;
            float3 n = quat_rotate_d(qr, FACE_NORMALS_D[f]);
            float3 cl = make_float3(FACE_NORMALS_D[f].x*0.5f*sc.x, FACE_NORMALS_D[f].y*0.5f*sc.y, FACE_NORMALS_D[f].z*0.5f*sc.z);
            float3 cWld = quat_rotate_d(qr, cl);
            cWld.x += tr.x; cWld.y += tr.y; cWld.z += tr.z;
            float3 tc = make_float3(cw.x-cWld.x, cw.y-cWld.y, cw.z-cWld.z);
            if (n.x*tc.x + n.y*tc.y + n.z*tc.z > 0.0f) { anyVisible = true; break; }
        }
        if (!anyVisible) return;

        int vi0 = CUBE_EDGES_D[edgeIdx][0], vi1 = CUBE_EDGES_D[edgeIdx][1];
        float3 lv0 = CUBE_VERTS_D[vi0], lv1 = CUBE_VERTS_D[vi1];
        float3 sv0 = make_float3(lv0.x*sc.x, lv0.y*sc.y, lv0.z*sc.z);
        float3 sv1 = make_float3(lv1.x*sc.x, lv1.y*sc.y, lv1.z*sc.z);
        float3 wv0 = quat_rotate_d(qr, sv0); wv0.x += tr.x; wv0.y += tr.y; wv0.z += tr.z;
        float3 wv1 = quat_rotate_d(qr, sv1); wv1.x += tr.x; wv1.y += tr.y; wv1.z += tr.z;

        int numSegs = 1 << subdiv;
        int vbase = atomicAdd(atomicVerts, numSegs * 4);
        int triBase = atomicAdd(atomicTris, numSegs * 2);
        if (!geometry_reservation_fits(vbase, numSegs * 4, triBase, numSegs * 2,
                                       atomicVerts, atomicTris))
            return;
        float img_w = camData[2], img_h = camData[3];
        float edgeWidth = get_wireframe_width(params);
        int edgeNdc = params.widthInNdc;
        float edgeScaling = params.depthScaling;
        uint32_t ec = pack_rgba8(0.784f, 0.784f, 0.784f);

        for (int seg = 0; seg < numSegs; seg++) {
            float ta = (float)seg / (float)numSegs;
            float tb = (float)(seg + 1) / (float)numSegs;
            float3 pa = make_float3(wv0.x + ta*(wv1.x-wv0.x), wv0.y + ta*(wv1.y-wv0.y), wv0.z + ta*(wv1.z-wv0.z));
            float3 pb = make_float3(wv0.x + tb*(wv1.x-wv0.x), wv0.y + tb*(wv1.y-wv0.y), wv0.z + tb*(wv1.z-wv0.z));
            clip_to_near(pa, pb, poseData, params.cullBehindCamera);
            float4 ca = ftheta_project(pa, poseData, camData);
            float4 cb = ftheta_project(pb, poseData, camData);
            float wa = fmaxf(fabsf(ca.w), 0.001f), wb = fmaxf(fabsf(cb.w), 0.001f);
            float ox, oy;
            get_edge_offset(ca, cb, wa, wb, img_w, img_h, edgeWidth, edgeNdc,
                            edgeScaling, ox, oy);
            float zba = -0.001f*ca.w, zbb = -0.001f*cb.w;
            int sv = vbase + seg * 4;
            int st = triBase + seg * 2;
            outVerts[sv+0]=make_float4(ca.x-ox*ca.w,ca.y-oy*ca.w,ca.z+zba,ca.w);
            outVerts[sv+1]=make_float4(ca.x+ox*ca.w,ca.y+oy*ca.w,ca.z+zba,ca.w);
            outVerts[sv+2]=make_float4(cb.x+ox*cb.w,cb.y+oy*cb.w,cb.z+zbb,cb.w);
            outVerts[sv+3]=make_float4(cb.x-ox*cb.w,cb.y-oy*cb.w,cb.z+zbb,cb.w);
            outIndices[st*3+0]=sv; outIndices[st*3+1]=sv+1; outIndices[st*3+2]=sv+2;
            outIndices[(st+1)*3+0]=sv; outIndices[(st+1)*3+1]=sv+2; outIndices[(st+1)*3+2]=sv+3;
            for (int i = 0; i < 4; i++) outVertColors[sv+i] = ec;
        }
    }
}

//------------------------------------------------------------------------
// Downsample kernel: 2x2 box filter for SSAA.
// Reads from hi-res RGBA8 buffer (ssW x ssH), writes to output (width x height).
//------------------------------------------------------------------------

__global__ void downsampleKernel(
    const uint8_t* __restrict__ hiRes,
    uint8_t* __restrict__ output,
    int width, int height,
    int ssW, int ssH,
    int camIdx)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int sx = px * 2;
    int sy = py * 2;

    uint32_t r = 0, g = 0, b = 0, a = 0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int hi = ((sy + dy) * ssW + (sx + dx)) * 4;
            r += hiRes[hi + 0];
            g += hiRes[hi + 1];
            b += hiRes[hi + 2];
            a += hiRes[hi + 3];
        }
    }

    int outIdx = (camIdx * height + py) * width + px;
    output[outIdx * 4 + 0] = (uint8_t)(r >> 2);
    output[outIdx * 4 + 1] = (uint8_t)(g >> 2);
    output[outIdx * 4 + 2] = (uint8_t)(b >> 2);
    output[outIdx * 4 + 3] = (uint8_t)(a >> 2);
}

//------------------------------------------------------------------------
// Fragment kernel: maps CudaRaster output (triangle IDs) to RGBA8.
//------------------------------------------------------------------------

// Fragment kernel with per-pixel barycentric interpolation.
// Reconstructs barycentrics from triangle vertex positions (same approach as nvdiffrast).
__global__ void fragmentKernel(
    const uint32_t* __restrict__ crColorBuffer,
    const int* __restrict__ indexBuffer,
    const float4* __restrict__ vertexPositions,
    const uint32_t* __restrict__ vertexColors,
    uint8_t* __restrict__ output,
    int width, int height, int crWidth, int crHeight,
    int camIdx,
    float depthScaling, int cameraTypeId, int noDepthFade)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int outIdx = (camIdx * height + py) * width + px;

    // CudaRaster uses bottom-up Y
    int cr_py = (height - 1) - py;
    int crIdx = px + crWidth * cr_py;
    uint32_t triId = crColorBuffer[crIdx];

    if (triId == 0) {
        output[outIdx * 4 + 0] = 0;
        output[outIdx * 4 + 1] = 0;
        output[outIdx * 4 + 2] = 0;
        output[outIdx * 4 + 3] = 0;
        return;
    }

    int triIdx = (int)triId - 1;

    int vi0 = indexBuffer[triIdx * 3 + 0];
    int vi1 = indexBuffer[triIdx * 3 + 1];
    int vi2 = indexBuffer[triIdx * 3 + 2];

    float4 p0 = vertexPositions[vi0];
    float4 p1 = vertexPositions[vi1];
    float4 p2 = vertexPositions[vi2];

    // Pixel center in NDC (CudaRaster's bottom-up coordinate system)
    float fx = (2.0f * px + 1.0f) / (float)crWidth - 1.0f;
    float fy = (2.0f * cr_py + 1.0f) / (float)crHeight - 1.0f;

    // Edge functions for perspective-correct barycentrics (nvdiffrast approach)
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;

    float a0 = p1x * p2y - p1y * p2x;
    float a1 = p2x * p0y - p2y * p0x;
    float a2 = p0x * p1y - p0y * p1x;

    float iw = 1.0f / (a0 + a1 + a2);
    float b0 = __saturatef(a0 * iw);
    float b1 = __saturatef(a1 * iw);
    float bs = 1.0f / fmaxf(b0 + b1, 1.0f);
    b0 *= bs;
    b1 *= bs;
    float b2 = 1.0f - b0 - b1;

    // Interpolate vertex colors
    uint32_t c0 = vertexColors[vi0];
    uint32_t c1 = vertexColors[vi1];
    uint32_t c2 = vertexColors[vi2];

    float r = b0 * (float)((c0 >>  0) & 0xFF) + b1 * (float)((c1 >>  0) & 0xFF) + b2 * (float)((c2 >>  0) & 0xFF);
    float g = b0 * (float)((c0 >>  8) & 0xFF) + b1 * (float)((c1 >>  8) & 0xFF) + b2 * (float)((c2 >>  8) & 0xFF);
    float b = b0 * (float)((c0 >> 16) & 0xFF) + b1 * (float)((c1 >> 16) & 0xFF) + b2 * (float)((c2 >> 16) & 0xFF);
    float a = b0 * (float)((c0 >> 24) & 0xFF) + b1 * (float)((c1 >> 24) & 0xFF) + b2 * (float)((c2 >> 24) & 0xFF);

    // Depth-based fog: darken with distance (matches GL fragment shader)
    // Disabled for BEV cameras and when depthScaling is off
    float z_interp = b0 * p0.z + b1 * p1.z + b2 * p2.z;
    // A colour dims towards black with distance unless the caller says not to,
    // which one matching a renderer that dims nothing needs. Line widths still
    // narrow with distance either way; the two are asked for separately.
    float fog = (depthScaling > 0.5f && cameraTypeId != 1 && !noDepthFade)
        ? fminf(fmaxf((1.0f - z_interp) * 0.5f, 0.0f), 1.0f) : 1.0f;
    r *= fog;
    g *= fog;
    b *= fog;

    output[outIdx * 4 + 0] = (uint8_t)fminf(r + 0.5f, 255.0f);
    output[outIdx * 4 + 1] = (uint8_t)fminf(g + 0.5f, 255.0f);
    output[outIdx * 4 + 2] = (uint8_t)fminf(b + 0.5f, 255.0f);
    output[outIdx * 4 + 3] = (uint8_t)fminf(a + 0.5f, 255.0f);
}

//------------------------------------------------------------------------
// Init / Destroy
//------------------------------------------------------------------------

void ludusCudaInit(NVDR_CTX_ARGS, LudusCudaState& s)
{
    memset(&s, 0, sizeof(LudusCudaState));
    s.cr = new CR::CudaRaster();
    s.tessellationThreshold = 0.0f;
    s.depthScaling = 1.0f;
    s.cullRadiusScale = 1.5f;
    s.resolutionScale = 1.0f;
    s.maxTessPolyline = 4;
    s.maxTessPolygon = 3;
    s.maxTessCube = 3;

    CUDA_CHECK(cudaMalloc(&s.atomicVertexCount, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.atomicTriangleCount, 2 * sizeof(int)));

    printf("LudusCuda: Initialized renderer\n");
}

void ludusCudaDestroy(NVDR_CTX_ARGS, LudusCudaState& s)
{
    if (s.cr) { delete s.cr; s.cr = nullptr; }
    if (s.projectedVertices) cudaFree(s.projectedVertices);
    if (s.triangleIndices) cudaFree(s.triangleIndices);
    if (s.vertexColors) cudaFree(s.vertexColors);
    if (s.triangleRanges) cudaFree(s.triangleRanges);
    if (s.atomicVertexCount) cudaFree(s.atomicVertexCount);
    if (s.atomicTriangleCount) cudaFree(s.atomicTriangleCount);
    if (s.outputBuffer) cudaFree(s.outputBuffer);
    if (s.colorPalette) cudaFree(s.colorPalette);
    if (s.widthTable) cudaFree(s.widthTable);
    if (s.msaaBuffer) cudaFree(s.msaaBuffer);
    memset(&s, 0, sizeof(LudusCudaState));
}

void ludusCudaUploadColorPalette(LudusCudaState& s, const uint32_t* hostPalette, int count)
{
    if (s.colorPalette) { cudaFree(s.colorPalette); s.colorPalette = nullptr; }
    s.colorPaletteSize = 0;
    if (count > 0 && hostPalette) {
        CUDA_CHECK(cudaMalloc(&s.colorPalette, count * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(s.colorPalette, hostPalette, count * sizeof(uint32_t), cudaMemcpyHostToDevice));
        s.colorPaletteSize = count;
    }
}

void ludusCudaUploadWidthTable(LudusCudaState& s, const float* hostWidths, int count)
{
    if (s.widthTable) { cudaFree(s.widthTable); s.widthTable = nullptr; }
    s.widthTableSize = 0;
    if (count > 0 && hostWidths) {
        CUDA_CHECK(cudaMalloc(&s.widthTable, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(s.widthTable, hostWidths, count * sizeof(float), cudaMemcpyHostToDevice));
        s.widthTableSize = count;
    }
}

//------------------------------------------------------------------------
// Ensure geometry buffers are large enough.

// Counts are taken as 64-bit because the estimates that produce them are
// products of scene-sized terms, and in `int` those wrap: a negative count
// sign-extends through the (size_t) casts below into a request of some
// exabytes, which cudaMalloc reports as the card being out of memory on a card
// that is doing nothing.
static void ensureBuffers(LudusCudaState& s, int64_t maxVerts, int64_t maxTris)
{
    if (maxVerts <= s.allocatedVertices && maxTris <= s.allocatedTriangles)
        return;

    if (s.projectedVertices) cudaFree(s.projectedVertices);
    if (s.triangleIndices) cudaFree(s.triangleIndices);
    if (s.vertexColors) cudaFree(s.vertexColors);
    s.projectedVertices = nullptr;
    s.triangleIndices = nullptr;
    s.vertexColors = nullptr;

    // The kernels index these buffers with int and compare against int atomic
    // counters, so a capacity past INT_MAX cannot be honoured whatever the card
    // holds.
    int64_t verts = maxVerts * 2;
    int64_t tris = maxTris * 2;
    if (verts > INT_MAX) verts = INT_MAX;
    if (tris > INT_MAX) tris = INT_MAX;

    s.allocatedVertices = (int)verts;
    s.allocatedTriangles = (int)tris;
    s.maxVertices = s.allocatedVertices;
    s.maxTriangles = s.allocatedTriangles;

    size_t vertBytes = (size_t)s.allocatedVertices * sizeof(float4);
    size_t triBytes  = (size_t)s.allocatedTriangles * 3 * sizeof(int);
    size_t colBytes  = (size_t)s.allocatedVertices * sizeof(uint32_t);

    // Debug: uncomment to log buffer allocations
    // fprintf(stderr, "[LudusCuda] ensureBuffers: verts=%d (%zu MB), tris=%d (%zu MB)\n",
    //         s.allocatedVertices, vertBytes / (1024*1024),
    //         s.allocatedTriangles, triBytes / (1024*1024));

    cudaError_t status = cudaMalloc(&s.projectedVertices, vertBytes);
    if (status == cudaSuccess) status = cudaMalloc(&s.triangleIndices, triBytes);
    if (status == cudaSuccess) status = cudaMalloc(&s.vertexColors, colBytes);

    // A capacity of zero against buffers that are not there is what keeps the
    // geometry kernels from writing through them: they bound every write by
    // these counters, so the draw reports an overflow and the camera comes out
    // blank rather than the process dying in the rasterizer.
    if (status != cudaSuccess) {
        fprintf(stderr,
                "[LudusCuda] no room for geometry buffers of %d vertices and %d "
                "triangles (%zu MB): %s\n",
                s.allocatedVertices, s.allocatedTriangles,
                (vertBytes + triBytes + colBytes) / (1024 * 1024),
                cudaGetErrorString(status));
        if (s.projectedVertices) cudaFree(s.projectedVertices);
        if (s.triangleIndices) cudaFree(s.triangleIndices);
        s.projectedVertices = nullptr;
        s.triangleIndices = nullptr;
        s.vertexColors = nullptr;
        s.allocatedVertices = 0;
        s.allocatedTriangles = 0;
        s.maxVertices = 0;
        s.maxTriangles = 0;
    }

    CUDA_CHECK(cudaMemcpy(s.atomicVertexCount + 1, &s.maxVertices,
                          sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.atomicTriangleCount + 1, &s.maxTriangles,
                          sizeof(int), cudaMemcpyHostToDevice));
}

// What an estimate may ask for up front, in triangles.
//
// An estimate is a first guess and is treated as one. A draw that comes up
// short sets the overflow flag, and the retry that follows asks for the counts
// the kernels actually produced, which then stand for the rest of the run. So
// a guess trimmed to this costs one redraw, while an untrimmed guess costs
// whatever the arithmetic happens to produce: the terms below multiply the
// varrays live at a single timestamp by the number of pools, and a map
// uploaded at one timestamp -- which is how a static scene is stamped -- puts
// the entire map in that first factor. Measured against a clip that tripped
// this, the guess was 1.88 billion triangles a camera and the kernels drew
// 654 thousand. A million is about 144 MB of buffers once doubled, and covers
// the densest map clips measured here.
static const int64_t ESTIMATE_TRIANGLE_CEILING = 1024 * 1024;

// How many times a draw may grow its buffers and start the camera again.
//
// One pass is usually enough, the counters carrying the whole demand because a
// thread reserves before it learns the reservation did not fit. A kernel that
// abandons the rest of its work on the first miss reports less than it wanted,
// though, and then one pass sizes the buffers from an undercount and the next
// misses again. Passes are cheap against a blank camera, so a few are allowed.
static const int GEOMETRY_RETRY_BUDGET = 3;

// Buffers for an estimated triangle count, vertices following at three a
// triangle as the geometry kernels emit them.
static void estimateBuffers(LudusCudaState& s, int64_t maxTris)
{
    if (maxTris > ESTIMATE_TRIANGLE_CEILING) maxTris = ESTIMATE_TRIANGLE_CEILING;
    if (maxTris < 1024) maxTris = 1024;
    ensureBuffers(s, maxTris * 3, maxTris);
}

// Whether a picture can be drawn at twice its size and averaged back down.
//
// The rasterizer maps a triangle's clip coordinates onto the viewport it is
// given and the offset only says where on the surface that viewport lands, so
// a picture too big for one viewport cannot be drawn a tile at a time: every
// tile would hold the whole picture, squeezed. A larger viewport writes
// outside its buffers rather than failing, so such a picture is drawn at its
// own size and left unsmoothed.
static bool supersamples(const LudusCudaState& s, int width, int height)
{
    if (s.msaaSamples < 4)
        return false;
    if (width * 2 <= CR_MAXVIEWPORT_SIZE && height * 2 <= CR_MAXVIEWPORT_SIZE)
        return true;

    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr,
                "[LudusCuda] %dx%d cannot be supersampled, twice it being past the "
                "rasterizer's limit of %d a side; drawing it unsmoothed\n",
                width, height, CR_MAXVIEWPORT_SIZE);
    }
    return false;
}

//------------------------------------------------------------------------
// Main render function.
//------------------------------------------------------------------------

void ludusCudaRender(
    NVDR_CTX_ARGS,
    LudusCudaState& s,
    cudaStream_t stream,
    const PolylineHeader* polylineHeaders, int numPolylines,
    const PolygonHeader* polygonHeaders, int numPolygons,
    const Cube* cubes, int numCubes,
    const Vertex* vertices, int numVertices,
    const Triangle* triangles, int numTriangles,
    const FThetaCamera* cameras, const CameraPose* poses,
    int numCameras, int width, int height,
    uint8_t* outputPtr,
    const CubePoolParams* cubePools, int numCubePools,
    const DotParams* dots, int numDotGroups)
{
    if (numCameras == 0 || width == 0 || height == 0) return;

    // Count total pool cubes and dots for geometry budget
    int totalPoolCubes = 0;
    for (int p = 0; p < numCubePools; p++)
        totalPoolCubes += cubePools[p].numCubes;
    int totalDots = 0;
    for (int d = 0; d < numDotGroups; d++)
        totalDots += dots[d].numDots;

    // Tessellation multipliers for geometry budget
    bool hasTess = s.tessellationThreshold > 0.0f;
    int tessPolyMul = hasTess ? (1 << (2 * s.maxTessPolygon)) : 1;
    int tessLineMul = hasTess ? (1 << s.maxTessPolyline) : 1;
    int tessCubeMul = hasTess ? (1 << (2 * s.maxTessCube)) : 1;

    // Estimate worst-case per-camera geometry
    int64_t maxTrisPerCam = (int64_t)numTriangles * tessPolyMul  // polygon triangles
                      + (int64_t)numCubes * 12         // immediate-mode cube faces (2 per face)
                      + (int64_t)numCubes * 24         // immediate-mode cube wireframe edges
                      + (int64_t)totalPoolCubes * 12 * tessCubeMul  // pool cube faces
                      + (int64_t)totalPoolCubes * 24 * tessCubeMul  // pool cube wireframe edges
                      + (int64_t)totalDots * 6         // dot hexagons (6 tris each)
                      + (int64_t)numPolylines * 512 * tessLineMul;  // polyline segments

    estimateBuffers(s, maxTrisPerCam);

    // SSAA: draw at twice the size and average when the size allows.
    bool smoothed = supersamples(s, width, height);
    int ssW = width, ssH = height;
    if (smoothed) {
        ssW = width * 2;
        ssH = height * 2;
    }

    int crWidth  = (ssW  + 7) & ~7;
    int crHeight = (ssH + 7) & ~7;
    int maxVp = (crWidth > crHeight) ? crWidth : crHeight;

    s.cr->setBufferSize(crWidth, crHeight, 1);

    // Allocate hi-res intermediate buffer for SSAA
    if (smoothed) {
        int needed = 4 * ssW * ssH;
        if (s.msaaBufferSize < needed) {
            if (s.msaaBuffer) cudaFree(s.msaaBuffer);
            CUDA_CHECK(cudaMalloc(&s.msaaBuffer, needed));
            s.msaaBufferSize = needed;
        }
    }

    // Render each camera sequentially
    int geometryRetryCount = 0;
    for (int camIdx = 0; camIdx < numCameras; camIdx++)
    {
        // Clear atomic counters
        CUDA_CHECK(cudaMemsetAsync(s.atomicVertexCount, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(s.atomicTriangleCount, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(s.atomicVertexCount + 2, 0, sizeof(int), stream));

        // Camera data pointers (raw float arrays matching struct layout)
        const float* camData  = (const float*)&cameras[camIdx];   // 18 floats
        const float* poseData = (const float*)&poses[camIdx];     // 16 floats

        // === Geometry generation ===

        if (numPolygons > 0) {
            polygonGeometryKernel<<<numPolygons, 64, 0, stream>>>(
                polygonHeaders, numPolygons, vertices, triangles,
                camData, poseData, s.tessellationThreshold, s.maxTessPolygon,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        if (numCubes > 0) {
            cubeGeometryKernel<<<numCubes, 6, 0, stream>>>(
                cubes, numCubes, camData, poseData, s.cullBehindCamera,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
            cubeWireframeKernel<<<numCubes, 12, 0, stream>>>(
                cubes, numCubes, camData, poseData,
                wireframe_width_of(s), s.widthInNdc, s.depthScaling, s.cullBehindCamera,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Timestamped cube pools (fused geometry + wireframe, 8 cubes/block)
        for (int p = 0; p < numCubePools; p++) {
            const CubePoolParams& pool = cubePools[p];
            if (pool.numCubes <= 0) continue;
            int nBlocks = (pool.numCubes + CUBES_PER_BLOCK - 1) / CUBES_PER_BLOCK;
            cubePoolFusedKernel<<<nBlocks, CUBE_POOL_BLOCK_SIZE, 0, stream>>>(
                pool.trackTimestamps, pool.prefixSum,
                pool.translations, pool.quaternions,
                pool.scales, pool.colors,
                pool.numCubes, pool.queryTimestampUs, pool.maxExtrapolationUs,
                pool.renderFlags,
                wireframe_width_of(s), s.widthInNdc, s.depthScaling, s.cullBehindCamera,
                camData, poseData, s.tessellationThreshold, s.maxTessCube,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        if (numPolylines > 0) {
            polylineGeometryKernel<<<numPolylines, 1, 0, stream>>>(
                polylineHeaders, numPolylines, vertices,
                camData, poseData, s.tessellationThreshold, s.maxTessPolyline, s.widthInNdc,
                s.cullBehindCamera,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Dot primitives (screen-space hexagons)
        for (int d = 0; d < numDotGroups; d++) {
            const DotParams& dg = dots[d];
            if (dg.numDots <= 0) continue;
            int nBlk = (dg.numDots + 127) / 128;
            dotGeometryKernel<<<nBlk, 128, 0, stream>>>(
                dg.positions, dg.numDots, dg.radius, dg.color,
                camData, poseData,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Read back actual counts
        int actualVerts = 0, actualTris = 0, geometryOverflow = 0;
        CUDA_CHECK(cudaMemcpyAsync(&actualTris, s.atomicTriangleCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&actualVerts, s.atomicVertexCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&geometryOverflow, s.atomicVertexCount + 2, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (geometryOverflow) {
            fprintf(stderr,
                    "[LudusCuda] geometry capacity exceeded (%d/%d vertices, %d/%d triangles); "
                    "growing buffers\n",
                    actualVerts, s.maxVertices, actualTris, s.maxTriangles);
            int64_t requiredVerts = actualVerts > s.maxVertices ? actualVerts : (int64_t)s.maxVertices + 1;
            int64_t requiredTris = actualTris > s.maxTriangles ? actualTris : (int64_t)s.maxTriangles + 1;
            ensureBuffers(s, requiredVerts, requiredTris);
            if (geometryRetryCount++ < GEOMETRY_RETRY_BUDGET) {
                --camIdx;
                continue;
            }
            fprintf(stderr, "[LudusCuda] geometry still overflowed after retry; zeroing camera output\n");
            CUDA_CHECK(cudaMemsetAsync(
                outputPtr + (size_t)camIdx * height * width * 4, 0,
                (size_t)width * height * 4, stream));
            geometryRetryCount = 0;
            continue;
        }
        geometryRetryCount = 0;

        // Debug: uncomment to log per-camera geometry counts and vertex data
        // fprintf(stderr, "[LudusCuda] cam %d: %d verts, %d tris (buf: %dx%d, cr: %dx%d)\n",
        //         camIdx, actualVerts, actualTris, width, height, crWidth, crHeight);
        // if (actualVerts > 0) {
        //     float4 dbgVerts[9];
        //     int dbgCount = actualVerts < 9 ? actualVerts : 9;
        //     cudaMemcpy(dbgVerts, s.projectedVertices, dbgCount * sizeof(float4), cudaMemcpyDeviceToHost);
        //     for (int i = 0; i < dbgCount; i++)
        //         fprintf(stderr, "  v[%d] = (%.4f, %.4f, %.4f, %.4f)\n", i, dbgVerts[i].x, dbgVerts[i].y, dbgVerts[i].z, dbgVerts[i].w);
        // }

        if (actualTris == 0) {
            // Clear this camera's output region
            CUDA_CHECK(cudaMemsetAsync(
                outputPtr + (size_t)camIdx * height * width * 4, 0,
                (size_t)width * height * 4, stream));
            continue;
        }

        // === CudaRaster ===
        s.cr->setVertexBuffer(s.projectedVertices, actualVerts);
        s.cr->setIndexBuffer(s.triangleIndices, actualTris);
        s.cr->setTiebreakerColorBuffer(s.vertexColors);
        s.cr->setDeterministicTiebreaker(true);
        s.cr->deferredClear(0);

        int tilesX = (crWidth  + maxVp - 1) / maxVp;
        int tilesY = (crHeight + maxVp - 1) / maxVp;
        bool rasterOk = true;

        for (int ty = 0; ty < tilesY; ty++) {
            for (int tx = 0; tx < tilesX; tx++) {
                int vpW = (tx < tilesX - 1) ? maxVp : (crWidth  - tx * maxVp);
                int vpH = (ty < tilesY - 1) ? maxVp : (crHeight - ty * maxVp);
                s.cr->setViewport(vpW, vpH, tx * maxVp, ty * maxVp);
                if (!s.cr->drawTriangles(nullptr, false, stream)) {
                    fprintf(stderr,
                            "[LudusCuda] cam %d tile (%d,%d): CudaRaster overflow; zeroing camera output\n",
                            camIdx, tx, ty);
                    CUDA_CHECK(cudaMemsetAsync(
                        outputPtr + (size_t)camIdx * height * width * 4, 0,
                        (size_t)width * height * 4, stream));
                    rasterOk = false;
                    break;
                }
            }
            if (!rasterOk)
                break;
        }
        if (!rasterOk)
            continue;

        // === Fragment pass (barycentric interpolation) ===
        const uint32_t* crColor = (const uint32_t*)s.cr->getColorBuffer();

        if (smoothed) {
            dim3 fragGrid((ssW + 7) / 8, (ssH + 7) / 8);
            dim3 fragBlock(8, 8);
            fragmentKernel<<<fragGrid, fragBlock, 0, stream>>>(
                crColor, s.triangleIndices,
                (const float4*)s.projectedVertices, s.vertexColors,
                s.msaaBuffer,
                ssW, ssH, crWidth, crHeight, 0,
                s.depthScaling, 0, s.noDepthFade);
            dim3 dsGrid((width + 7) / 8, (height + 7) / 8);
            downsampleKernel<<<dsGrid, dim3(8, 8), 0, stream>>>(
                s.msaaBuffer, outputPtr,
                width, height, ssW, ssH, camIdx);
        } else {
            dim3 fragGrid((width + 7) / 8, (height + 7) / 8);
            dim3 fragBlock(8, 8);
            fragmentKernel<<<fragGrid, fragBlock, 0, stream>>>(
                crColor, s.triangleIndices,
                (const float4*)s.projectedVertices, s.vertexColors,
                outputPtr,
                width, height, crWidth, crHeight, camIdx,
                s.depthScaling, 0, s.noDepthFade);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

//------------------------------------------------------------------------
// Timestamped render entry point.
// Accepts flat buffers (same layout as GL SSBOs) and pool headers.
// All timestamp search, element extraction, and geometry generation on GPU.
//------------------------------------------------------------------------

void ludusCudaRenderTimestamped(
    NVDR_CTX_ARGS,
    LudusCudaState& s,
    cudaStream_t stream,
    const int64_t* timestamps,
    const int32_t* int32Data,
    const Vertex* vertices,
    const Triangle* triangles,
    const float* floatData,
    const TsPolylinePoolHeader* polylinePools, int numPolylinePools,
    const TsPolygonPoolHeader* polygonPools, int numPolygonPools,
    const TsCubePoolHeader* cubePools, int numCubePools,
    int totalCubes, const int64_t* cubePoolCounts,
    const int64_t* queryTimestampsUs, int maxExtrapolationUs,
    int maxVarraysPerTsPolyline, int maxVarraysPerTsPolygon,
    const CudaRenderParams& params,
    const FThetaCamera* cameras, const CameraPose* poses,
    int numCameras, int width, int height,
    uint8_t* outputPtr)
{
    if (numCameras == 0 || width == 0 || height == 0) return;

    // Pool headers stay device-resident; totalCubes is cached at scene upload.
    bool hasTess = params.tessellationThreshold > 0.0f;
    int tessPolyMul = hasTess ? (1 << (2 * params.maxTessPolygon)) : 1;
    int tessLineMul = hasTess ? (1 << params.maxTessPolyline) : 1;
    int tessCubeMul = hasTess ? (1 << (2 * params.maxTessCube)) : 1;

    // Compute geometry budget per camera using actual per-timestamp max varrays
    // Use per-timestamp max (computed from prefix sums on Python side)
    int mvPl = maxVarraysPerTsPolyline;
    int mvPg = maxVarraysPerTsPolygon;
    if (mvPl < 1) mvPl = 1;
    if (mvPg < 1) mvPg = 1;

    int64_t maxTrisPerCam = (int64_t)mvPl * numPolylinePools * 512 * tessLineMul
                      + (int64_t)mvPg * numPolygonPools  * 64  * tessPolyMul
                      + (int64_t)totalCubes * 12 * tessCubeMul
                      + (int64_t)totalCubes * 24 * tessCubeMul
                      + (int64_t)mvPl * numPolylinePools * 42;

    // Debug: uncomment to log geometry budget
    // fprintf(stderr, "[LudusCudaTS] budget: mvPl=%d mvPg=%d totalCubes=%d maxTris=%lld\n",
    //         mvPl, mvPg, totalCubes, (long long)maxTrisPerCam);

    estimateBuffers(s, maxTrisPerCam);

    // SSAA: draw at twice the size and average when the size allows.
    bool smoothed = supersamples(s, width, height);
    int ssW = width, ssH = height;
    if (smoothed) {
        ssW = width * 2;
        ssH = height * 2;
    }

    int crWidth  = (ssW  + 7) & ~7;
    int crHeight = (ssH + 7) & ~7;
    int maxVp = (crWidth > crHeight) ? crWidth : crHeight;
    s.cr->setBufferSize(crWidth, crHeight, 1);

    // Allocate hi-res intermediate buffer for SSAA
    if (smoothed) {
        int needed = 4 * ssW * ssH;
        if (s.msaaBufferSize < needed) {
            if (s.msaaBuffer) cudaFree(s.msaaBuffer);
            CUDA_CHECK(cudaMalloc(&s.msaaBuffer, needed));
            s.msaaBufferSize = needed;
        }
    }

    int geometryRetryCount = 0;
    for (int camIdx = 0; camIdx < numCameras; camIdx++)
    {
        int64_t queryTimestampUs = queryTimestampsUs[camIdx];
        CUDA_CHECK(cudaMemsetAsync(s.atomicVertexCount, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(s.atomicTriangleCount, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(s.atomicVertexCount + 2, 0, sizeof(int), stream));

        const float* camData  = (const float*)&cameras[camIdx];
        const float* poseData = (const float*)&poses[camIdx];

        // Polygon pools
        if (numPolygonPools > 0 && mvPg > 0) {
            int nBlocks = numPolygonPools * mvPg;
            polygonPoolKernel<<<nBlocks, 64, 0, stream>>>(
                polygonPools, numPolygonPools,
                timestamps, int32Data, vertices, triangles, floatData,
                queryTimestampUs, mvPg,
                camData, poseData, params,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Cube pools (flat-buffer variant)
        for (int p = 0; p < numCubePools; p++) {
            // Pool sizes are captured on the host during scene upload, so
            // dispatch exact grids without a per-frame device readback.
            int nc = (int)cubePoolCounts[p];
            if (nc <= 0) continue;
            int nCubeBlocks = (nc + CUBES_PER_BLOCK - 1) / CUBES_PER_BLOCK;
            cubePoolFlatKernel<<<dim3(nCubeBlocks, 1), CUBE_POOL_BLOCK_SIZE, 0, stream>>>(
                cubePools + p, 1,
                timestamps, int32Data, floatData,
                queryTimestampUs, maxExtrapolationUs,
                camData, poseData, params,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Polyline pools (including dots)
        if (numPolylinePools > 0 && mvPl > 0) {
            int nBlocks = numPolylinePools * mvPl;
            polylinePoolKernel<<<nBlocks, 1, 0, stream>>>(
                polylinePools, numPolylinePools,
                timestamps, int32Data, vertices, floatData,
                queryTimestampUs, mvPl,
                camData, poseData, params,
                (float4*)s.projectedVertices, s.triangleIndices, s.vertexColors,
                s.atomicVertexCount, s.atomicTriangleCount);
        }

        // Read back actual counts
        int actualVerts = 0, actualTris = 0, geometryOverflow = 0;
        CUDA_CHECK(cudaMemcpyAsync(&actualTris, s.atomicTriangleCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&actualVerts, s.atomicVertexCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&geometryOverflow, s.atomicVertexCount + 2, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (geometryOverflow) {
            fprintf(stderr,
                    "[LudusCudaTS] geometry capacity exceeded (%d/%d vertices, %d/%d triangles); "
                    "growing buffers\n",
                    actualVerts, s.maxVertices, actualTris, s.maxTriangles);
            int64_t requiredVerts = actualVerts > s.maxVertices ? actualVerts : (int64_t)s.maxVertices + 1;
            int64_t requiredTris = actualTris > s.maxTriangles ? actualTris : (int64_t)s.maxTriangles + 1;
            ensureBuffers(s, requiredVerts, requiredTris);
            if (geometryRetryCount++ < GEOMETRY_RETRY_BUDGET) {
                --camIdx;
                continue;
            }
            fprintf(stderr, "[LudusCudaTS] geometry still overflowed after retry; zeroing camera output\n");
            CUDA_CHECK(cudaMemsetAsync(
                outputPtr + (size_t)camIdx * height * width * 4, 0,
                (size_t)width * height * 4, stream));
            geometryRetryCount = 0;
            continue;
        }
        geometryRetryCount = 0;

        // Debug: uncomment to log per-camera geometry counts
        // fprintf(stderr, "[LudusCudaTS] cam %d: %d verts, %d tris (cr: %dx%d)\n",
        //         camIdx, actualVerts, actualTris, crWidth, crHeight);

        if (actualTris == 0) {
            CUDA_CHECK(cudaMemsetAsync(
                outputPtr + (size_t)camIdx * height * width * 4, 0,
                (size_t)width * height * 4, stream));
            continue;
        }

        // CudaRaster
        s.cr->setVertexBuffer(s.projectedVertices, actualVerts);
        s.cr->setIndexBuffer(s.triangleIndices, actualTris);
        s.cr->setTiebreakerColorBuffer(s.vertexColors);
        s.cr->setDeterministicTiebreaker(true);
        s.cr->deferredClear(0);

        int tilesX = (crWidth  + maxVp - 1) / maxVp;
        int tilesY = (crHeight + maxVp - 1) / maxVp;
        bool rasterOk = true;
        for (int ty = 0; ty < tilesY; ty++) {
            for (int tx = 0; tx < tilesX; tx++) {
                int vpW = (tx < tilesX - 1) ? maxVp : (crWidth  - tx * maxVp);
                int vpH = (ty < tilesY - 1) ? maxVp : (crHeight - ty * maxVp);
                s.cr->setViewport(vpW, vpH, tx * maxVp, ty * maxVp);
                if (!s.cr->drawTriangles(nullptr, false, stream)) {
                    fprintf(stderr,
                            "[LudusCudaTS] cam %d tile (%d,%d): CudaRaster overflow; zeroing camera output\n",
                            camIdx, tx, ty);
                    CUDA_CHECK(cudaMemsetAsync(
                        outputPtr + (size_t)camIdx * height * width * 4, 0,
                        (size_t)width * height * 4, stream));
                    rasterOk = false;
                    break;
                }
            }
            if (!rasterOk)
                break;
        }
        if (!rasterOk)
            continue;

        // Fragment pass (barycentric interpolation)
        const uint32_t* crColor = (const uint32_t*)s.cr->getColorBuffer();
        if (smoothed) {
            dim3 fragGrid((ssW + 7) / 8, (ssH + 7) / 8);
            dim3 fragBlock(8, 8);
            fragmentKernel<<<fragGrid, fragBlock, 0, stream>>>(
                crColor, s.triangleIndices,
                (const float4*)s.projectedVertices, s.vertexColors,
                s.msaaBuffer,
                ssW, ssH, crWidth, crHeight, 0,
                params.depthScaling, params.cameraTypeId, params.noDepthFade);
            dim3 dsGrid((width + 7) / 8, (height + 7) / 8);
            downsampleKernel<<<dsGrid, dim3(8, 8), 0, stream>>>(
                s.msaaBuffer, outputPtr,
                width, height, ssW, ssH, camIdx);
        } else {
            dim3 fragGrid((width + 7) / 8, (height + 7) / 8);
            dim3 fragBlock(8, 8);
            fragmentKernel<<<fragGrid, fragBlock, 0, stream>>>(
                crColor, s.triangleIndices,
                (const float4*)s.projectedVertices, s.vertexColors,
                outputPtr,
                width, height, crWidth, crHeight, camIdx,
                params.depthScaling, params.cameraTypeId, params.noDepthFade);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

//------------------------------------------------------------------------
