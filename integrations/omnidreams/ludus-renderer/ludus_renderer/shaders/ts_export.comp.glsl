// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#version 460

// The graphics pass resolves visibility into this cached Vulkan-only image.
// This compute pass is the sole writer of the CUDA-visible linear allocation,
// avoiding unordered fragment SSBO stores while retaining zero-copy interop.
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 14, std430) writeonly buffer OutputPixels {
    uint rgba8[];
} output_pixels;

layout(set = 0, binding = 15, rgba8) uniform readonly image2DArray color_image;

layout(push_constant) uniform PushConstants {
    float u_width_polyline_regular;
    float u_width_polyline_bev;
    float u_width_ego_traj_regular;
    float u_width_ego_traj_bev;
    float u_width_wireframe;
    float u_resolution_scale;
    float u_depth_scaling;
    int u_max_extrapolation_us;
    int u_color_palette_size;
    uint u_num_queries;
    float u_tessellation_threshold;
    uint u_max_tessellation_polyline;
    uint u_max_tessellation_polygon;
    uint u_max_tessellation_cube;
    float u_cull_radius_scale;
    float u_fog_enabled;
    uint u_max_obstacles;
    uint u_cube_pool_index;
    uint u_num_polygon_pools;
    uint u_max_varrays_per_pool;
    uint u_num_polyline_pools;
    uint u_output_width;
    uint u_output_height;
} pc;

void main() {
    uvec3 pixel = gl_GlobalInvocationID;
    if (pixel.x >= pc.u_output_width ||
        pixel.y >= pc.u_output_height ||
        pixel.z >= pc.u_num_queries) {
        return;
    }

    vec4 color = imageLoad(color_image, ivec3(pixel));
    uvec4 rgba = uvec4(clamp(color, 0.0, 1.0) * 255.0 + 0.5);
    uint pixel_index =
        (pixel.z * pc.u_output_height + pixel.y) * pc.u_output_width + pixel.x;
    output_pixels.rgba8[pixel_index] =
        rgba.r | (rgba.g << 8) | (rgba.b << 16) | (rgba.a << 24);
}
