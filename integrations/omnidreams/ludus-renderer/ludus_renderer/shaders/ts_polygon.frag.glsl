
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

// Per-primitive color block from mesh shader
layout(location = 0) perprimitiveEXT in PrimColorBlock {
    vec3 color;
    float is_bev;
} prim_in;

layout(location = 0) out vec4 out_color;


IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    vec3 color = prim_in.color;
    
    // Apply distance-based fog (darken with distance) - disabled for BEV cameras
    // gl_FragCoord.z is in [0, 1] where 0=near, 1=far
    if (pc.u_fog_enabled > 0.5 && prim_in.is_bev < 0.5) {
        float fog_factor = 1.0 - gl_FragCoord.z;  // 1.0 at near, 0.0 at far
        fog_factor = clamp(fog_factor, 0.0, 1.0);
        color *= fog_factor;
    }
    
    out_color = vec4(color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
