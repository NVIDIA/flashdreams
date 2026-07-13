
#version 460
#extension GL_EXT_fragment_shader_barycentric : require
#extension GL_EXT_mesh_shader : require

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
} pc;

#define IF_ZMODIFY(x)

// Per-primitive gradient data from mesh shader (no per-vertex outputs needed!)
layout(location = 0) perprimitiveEXT in CubeGradientBlock {
    vec3 corner_t;      // gradient t values at triangle's 3 corners
    vec3 front_color;
    vec3 back_color;
    float is_bev;       // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_in;

layout(location = 0) out vec4 out_color;


IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    // Interpolate gradient t using hardware barycentric coordinates
    float t = dot(prim_in.corner_t, gl_BaryCoordEXT);

    // Compute gradient color
    vec3 color = mix(prim_in.back_color, prim_in.front_color, t);

    // Apply distance-based fog (darken with distance) - disabled for BEV cameras
    if (pc.u_fog_enabled > 0.5 && prim_in.is_bev < 0.5) {
        float fog_factor = 1.0 - gl_FragCoord.z;
        fog_factor = clamp(fog_factor, 0.0, 1.0);
        color *= fog_factor;
    }

    out_color = vec4(color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
