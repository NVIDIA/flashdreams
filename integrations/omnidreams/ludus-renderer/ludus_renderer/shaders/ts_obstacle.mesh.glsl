#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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

// Cap styles
const uint CAP_NONE  = 0u;
const uint CAP_FLAT  = 1u;
const uint CAP_ROUND = 2u;

// Primitive type IDs (must match Python PRIM_* constants in ops.py)
const uint PRIM_ROAD_BOUNDARY    = 0u;
const uint PRIM_LANE_LINE        = 1u;   // Legacy cyan lane line
const uint PRIM_CROSSWALK        = 2u;
const uint PRIM_STATIC_OBSTACLE  = 3u;   // Legacy, avoid using
const uint PRIM_EGO_TRAJECTORY   = 4u;
const uint PRIM_OBSTACLE         = 5u;
const uint PRIM_EGO_OBSTACLE     = 6u;
const uint PRIM_WAIT_LINE        = 7u;
const uint PRIM_POLE             = 8u;
const uint PRIM_ROAD_MARKING     = 9u;
const uint PRIM_LANE_BOUNDARY    = 10u;
const uint PRIM_TRAFFIC_LIGHT    = 11u;
const uint PRIM_TRAFFIC_SIGN     = 12u;
const uint PRIM_INTERSECTION     = 13u;
const uint PRIM_ROAD_ISLAND      = 14u;
const uint PRIM_BUFFER_ZONE      = 15u;
// Lane line variants by color/style
const uint PRIM_LANE_LINE_WHITE_SOLID   = 16u;
const uint PRIM_LANE_LINE_WHITE_DASHED  = 17u;
const uint PRIM_LANE_LINE_YELLOW_SOLID  = 18u;
const uint PRIM_LANE_LINE_YELLOW_DASHED = 19u;
// Dot primitives - each vertex is rendered as a circle
const uint PRIM_DOT_YELLOW = 20u;
const uint PRIM_DOT_WHITE  = 21u;

// Camera type IDs
const uint CAMERA_TYPE_REGULAR = 0u;
const uint CAMERA_TYPE_BEV     = 1u;

//=============================================================================
// Hardcoded Primitive Styles (matching wm-render colorscheme v3)
//=============================================================================

// Colors (RGB normalized) - matching imaginaire4 v3 colorscheme
// Source: imaginaire4/imaginaire/auxiliary/world_scenario/color_scheme/config_color_hdmap.json
const vec3 COLOR_ROAD_BOUNDARY   = vec3(253.0/255.0, 1.0/255.0, 232.0/255.0);   // Magenta (253, 1, 232)
const vec3 COLOR_LANE_LINE       = vec3(98.0/255.0, 183.0/255.0, 249.0/255.0);  // Cyan (98, 183, 249) - "lanelines" (legacy)
const vec3 COLOR_CROSSWALK       = vec3(139.0/255.0, 93.0/255.0, 255.0/255.0);  // Purple (139, 93, 255)
const vec3 COLOR_STATIC_OBSTACLE = vec3(255.0/255.0, 100.0/255.0, 0.0/255.0);   // Orange (legacy)
const vec3 COLOR_EGO_TRAJECTORY  = vec3(0.0/255.0, 255.0/255.0, 0.0/255.0);     // Green (0, 255, 0)
const vec3 COLOR_WAIT_LINE       = vec3(108.0/255.0, 179.0/255.0, 59.0/255.0);  // Yellow-green (108, 179, 59)
const vec3 COLOR_POLE            = vec3(183.0/255.0, 69.0/255.0, 177.0/255.0);  // Purple-magenta (183, 69, 177)
const vec3 COLOR_ROAD_MARKING    = vec3(20.0/255.0, 254.0/255.0, 185.0/255.0);  // Cyan-green (20, 254, 185)
const vec3 COLOR_LANE_BOUNDARY   = vec3(98.0/255.0, 183.0/255.0, 249.0/255.0);  // Cyan (same as lane line)
const vec3 COLOR_TRAFFIC_LIGHT   = vec3(100.0/255.0, 100.0/255.0, 100.0/255.0); // Gray (100, 100, 100)
const vec3 COLOR_TRAFFIC_SIGN    = vec3(8.0/255.0, 2.0/255.0, 255.0/255.0);     // Blue (8, 2, 255)
const vec3 COLOR_INTERSECTION    = vec3(80.0/255.0, 80.0/255.0, 120.0/255.0);   // Dark blue-gray (approximated)
const vec3 COLOR_ROAD_ISLAND     = vec3(60.0/255.0, 120.0/255.0, 60.0/255.0);   // Dark green (approximated)
const vec3 COLOR_BUFFER_ZONE     = vec3(120.0/255.0, 80.0/255.0, 80.0/255.0);   // Dark red-brown (approximated)
// Lane line variants from config_color_geometry_laneline.json
const vec3 COLOR_LANE_LINE_WHITE  = vec3(255.0/255.0, 255.0/255.0, 255.0/255.0); // White (255, 255, 255)
const vec3 COLOR_LANE_LINE_YELLOW = vec3(255.0/255.0, 255.0/255.0, 0.0/255.0);   // Yellow (255, 255, 0)

// Default polyline widths (pixels at reference resolution 1280x720)
const float DEFAULT_WIDTH_POLYLINE_REGULAR = 7.0;
const float DEFAULT_WIDTH_POLYLINE_BEV     = 4.0;
const float DEFAULT_WIDTH_EGO_TRAJ_REGULAR = 12.0;
const float DEFAULT_WIDTH_EGO_TRAJ_BEV     = 5.0;
const float DEFAULT_WIDTH_POLE_REGULAR     = 5.0;   // Poles are thinner (reference: 5 vs 12)
const float DEFAULT_WIDTH_POLE_BEV         = 3.0;
const float DEFAULT_WIDTH_WIREFRAME        = 2.0;

// Configurable width uniforms (set from Python, or use defaults)

// Compute depth-based scaling factor for line width
// z_ndc is in [-1, 1] where -1 is near, +1 is far
// Returns 1.0 at near, 0.0 at far (linear fade)
float get_depth_scale(vec4 clip) {
    if (pc.u_depth_scaling < 0.5) return 1.0;  // Disabled
    float z_ndc = clip.z / clip.w;  // Convert to NDC
    float depth_scale = 1.0 - z_ndc;  // Map [-1,1] to [1,0]
    return clamp(depth_scale, 0.0, 1.0);
}

// Color palette buffer (optional, configured from Python)
// Declared early so get_prim_color can use it; actual buffer layout defined later
layout(std430, binding = 10) readonly buffer ColorPaletteBufferEarly {
    vec4 g_color_palette[];
};

// Get default color for primitive type (hardcoded fallback)
vec3 get_default_prim_color(uint prim_type_id) {
    if (prim_type_id == PRIM_ROAD_BOUNDARY)   return COLOR_ROAD_BOUNDARY;
    if (prim_type_id == PRIM_LANE_LINE)       return COLOR_LANE_LINE;
    if (prim_type_id == PRIM_CROSSWALK)       return COLOR_CROSSWALK;
    if (prim_type_id == PRIM_STATIC_OBSTACLE) return COLOR_STATIC_OBSTACLE;
    if (prim_type_id == PRIM_EGO_TRAJECTORY)  return COLOR_EGO_TRAJECTORY;
    if (prim_type_id == PRIM_WAIT_LINE)       return COLOR_WAIT_LINE;
    if (prim_type_id == PRIM_POLE)            return COLOR_POLE;
    if (prim_type_id == PRIM_ROAD_MARKING)    return COLOR_ROAD_MARKING;
    if (prim_type_id == PRIM_LANE_BOUNDARY)   return COLOR_LANE_BOUNDARY;
    if (prim_type_id == PRIM_TRAFFIC_LIGHT)   return COLOR_TRAFFIC_LIGHT;
    if (prim_type_id == PRIM_TRAFFIC_SIGN)    return COLOR_TRAFFIC_SIGN;
    if (prim_type_id == PRIM_INTERSECTION)    return COLOR_INTERSECTION;
    if (prim_type_id == PRIM_ROAD_ISLAND)     return COLOR_ROAD_ISLAND;
    if (prim_type_id == PRIM_BUFFER_ZONE)     return COLOR_BUFFER_ZONE;
    // Lane line variants
    if (prim_type_id == PRIM_LANE_LINE_WHITE_SOLID)   return COLOR_LANE_LINE_WHITE;
    if (prim_type_id == PRIM_LANE_LINE_WHITE_DASHED)  return COLOR_LANE_LINE_WHITE;
    if (prim_type_id == PRIM_LANE_LINE_YELLOW_SOLID)  return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_LANE_LINE_YELLOW_DASHED) return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_DOT_YELLOW) return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_DOT_WHITE)  return COLOR_LANE_LINE_WHITE;
    return vec3(1.0, 1.0, 1.0);  // Default white for obstacles
}

// Get color for primitive type (checks palette first, then falls back to defaults)
vec3 get_prim_color(uint prim_type_id) {
    // If color palette is configured, check if this prim_type has a custom color
    // Alpha < 0 means "use default", alpha >= 0 means "use this color"
    if (pc.u_color_palette_size > 0 && prim_type_id < uint(pc.u_color_palette_size)) {
        vec4 palette_color = g_color_palette[prim_type_id];
        if (palette_color.a >= 0.0) {
            return palette_color.rgb;
        }
    }
    // Fall back to hardcoded defaults
    return get_default_prim_color(prim_type_id);
}

// Check if primitive is a dot type (rendered as circles at each vertex)
bool is_dot_primitive(uint prim_type_id) {
    return prim_type_id == PRIM_DOT_YELLOW || prim_type_id == PRIM_DOT_WHITE;
}

// Get polyline width for primitive type and camera type (scaled by resolution)
float get_prim_width(uint prim_type_id, uint camera_type_id) {
    bool is_bev = (camera_type_id == CAMERA_TYPE_BEV);
    float scale = (pc.u_resolution_scale > 0.0) ? pc.u_resolution_scale : 1.0;

    float base_width;
    if (prim_type_id == PRIM_EGO_TRAJECTORY) {
        float default_w = is_bev ? DEFAULT_WIDTH_EGO_TRAJ_BEV : DEFAULT_WIDTH_EGO_TRAJ_REGULAR;
        float custom_w = is_bev ? pc.u_width_ego_traj_bev : pc.u_width_ego_traj_regular;
        base_width = (custom_w > 0.0) ? custom_w : default_w;
    } else if (prim_type_id == PRIM_POLE) {
        // Poles use thinner lines (reference: 5 pixels vs 12 for other polylines)
        base_width = is_bev ? DEFAULT_WIDTH_POLE_BEV : DEFAULT_WIDTH_POLE_REGULAR;
    } else {
        float default_w = is_bev ? DEFAULT_WIDTH_POLYLINE_BEV : DEFAULT_WIDTH_POLYLINE_REGULAR;
        float custom_w = is_bev ? pc.u_width_polyline_bev : pc.u_width_polyline_regular;
        base_width = (custom_w > 0.0) ? custom_w : default_w;
    }
    return base_width * scale;
}

// Get wireframe edge width (scaled by resolution)
float get_wireframe_width() {
    float scale = (pc.u_resolution_scale > 0.0) ? pc.u_resolution_scale : 1.0;
    float base_width = (pc.u_width_wireframe > 0.0) ? pc.u_width_wireframe : DEFAULT_WIDTH_WIREFRAME;
    return base_width * scale;
}

//=============================================================================
// Data Structures (must match C++ structs)
//=============================================================================

// TimestampedPolylinePool (64 bytes)
struct TimestampedPolylinePool {
    uint  num_timestamps;
    uint  num_varrays;
    uint  num_vertices;
    uint  prim_type_id;              // Index into PrimitiveStyle lookup table
    uint  timestamps_offset;
    uint  ts_varrays_ps_offset;
    uint  varrays_ps_offset;
    uint  vertices_offset;
    uint  aabb_offset;               // Per-element AABB in float buffer (6 floats each: min xyz, max xyz)
    uint  _pad1, _pad2, _pad3, _pad4, _pad5, _pad6, _pad7;  // Padding to 64 bytes
};

// TimestampedPolygonPool (64 bytes)
struct TimestampedPolygonPool {
    uint  num_timestamps;
    uint  num_varrays;
    uint  num_vertices;
    uint  num_triangles;
    uint  prim_type_id;              // Index into PrimitiveStyle lookup table
    uint  timestamps_offset;
    uint  ts_varrays_ps_offset;
    uint  varrays_ps_offset;
    uint  tri_ps_offset;
    uint  vertices_offset;
    uint  triangles_offset;
    uint  aabb_offset;               // Per-element AABB in float buffer (6 floats each: min xyz, max xyz)
    uint  _pad1, _pad2, _pad3, _pad4;  // Padding to 64 bytes (16 uints)
};

// CubePool (64 bytes) - used for obstacles, traffic lights, traffic signs, etc.
// Renamed from ObstaclePool but keeping struct name for backward compat
struct ObstaclePool {
    uint num_cubes;                  // Was: num_obstacles
    uint num_timestamps;
    uint num_track_poses;
    uint prim_type_id;               // Semantic type for color lookup
    uint timestamps_offset;          // Global timestamps for this pool
    uint cube_ts_ps_offset;          // Per-cube track length prefix sum (was: obstacle_ts_ps_offset)
    uint track_timestamps_offset;    // Per-pose timestamps in float buffer (as 2x uint for int64)
    uint translations_offset;        // Per-pose translations (3 floats each)
    uint quaternions_offset;         // Per-pose quaternions (4 floats each)
    uint scales_offset;              // Per-cube scales (3 floats each)
    uint colors_offset;              // Per-cube colors (6 floats each)
    uint render_flags;               // CUBE_FLAG_* bits (e.g., CUBE_FLAG_WIREFRAME)
    uint _pad2, _pad3, _pad4, _pad5; // 4 padding fields for 64 bytes total
};

// Cube render flags
const uint CUBE_FLAG_WIREFRAME = 1u;  // Draw wireframe edges in addition to solid faces

// TimestampedScene (128 bytes)
struct TimestampedScene {
    uint num_polyline_pools;
    uint polyline_pools_offset;
    uint num_polygon_pools;
    uint polygon_pools_offset;
    uint num_cube_pools;             // Was: has_obstacle_pool (0/1), now supports multiple
    uint cube_pools_offset;          // Was: obstacle_pool_offset
    uint timestamps_buffer_offset;
    uint int32_buffer_offset;
    uint vertex_buffer_offset;
    uint triangle_buffer_offset;
    uint pose_buffer_offset;
    uint float_buffer_offset;
    uint _pad[20];
};

// RenderQuery (16 bytes)
struct RenderQuery {
    uint   scene_id;
    uint   camera_id;
    int64_t timestamp_us;
    uint   camera_type_id;    // CAMERA_TYPE_REGULAR or CAMERA_TYPE_BEV
    uint   _pad1, _pad2, _pad3;  // Padding to 32 bytes
};

// FThetaCamera (72 bytes)
struct FThetaCamera {
    float cx, cy;
    float img_w, img_h;
    float poly0, poly1, poly2, poly3, poly4, poly5;
    float max_ray_angle;
    float max_distortion_val;
    float max_distortion_dval;
    float depth_max;
    float ld_c, ld_d, ld_e, ld_f;
};

// CameraPose (64 bytes)
struct CameraPose {
    mat4 world_to_camera;
};

// Vertex (16 bytes)
struct Vertex {
    float x, y, z;
    float _pad;
};

//=============================================================================
// Buffer Bindings
//=============================================================================

// Global data buffers
layout(std430, binding = 0) readonly buffer TimestampsBuffer {
    int64_t g_timestamps[];
};

layout(std430, binding = 1) readonly buffer Int32Buffer {
    int g_int32[];
};

layout(std430, binding = 2) readonly buffer VertexBuffer {
    Vertex g_vertices[];
};

layout(std430, binding = 3) readonly buffer TriangleBuffer {
    uvec3 g_triangles[];
};

layout(std430, binding = 4) readonly buffer PoseBuffer {
    CameraPose g_poses[];
};

layout(std430, binding = 5) readonly buffer FloatBuffer {
    float g_floats[];
};

// Scene metadata
layout(std430, binding = 6) readonly buffer SceneBuffer {
    TimestampedScene g_scenes[];
};

layout(std430, binding = 7) readonly buffer PolylinePoolBuffer {
    TimestampedPolylinePool g_polyline_pools[];
};

layout(std430, binding = 8) readonly buffer PolygonPoolBuffer {
    TimestampedPolygonPool g_polygon_pools[];
};

layout(std430, binding = 9) readonly buffer ObstaclePoolBuffer {
    ObstaclePool g_obstacle_pools[];
};

// Camera buffers
layout(std430, binding = 11) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera g_camera_intrinsics[];
};

layout(std430, binding = 12) readonly buffer CameraPoseBuffer {
    CameraPose g_camera_poses[];  // One per query (dynamic viewpoints)
};

// Query buffer
layout(std430, binding = 13) readonly buffer QueryBuffer {
    RenderQuery g_queries[];
};

// Uniforms
// Spatial culling: elements beyond depth_max * scale from the camera are
// discarded in the task shader.  Scale > 1 gives headroom so nothing at
// the visible boundary pops in/out.  Set to 0 to disable culling.

//=============================================================================
// GPU Binary Search
// Returns index of last element <= target, or -1 if target < all elements
//=============================================================================

int binary_search_timestamps(uint base_offset, uint count, int64_t target) {
    if (count == 0u) return -1;

    int left = 0;
    int right = int(count) - 1;
    int result = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int64_t val = g_timestamps[base_offset + uint(mid)];

        if (val <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

//=============================================================================
// Quaternion Math for Track Interpolation
//=============================================================================

// Quaternion dot product
float quat_dot(vec4 a, vec4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// Quaternion normalize
vec4 quat_normalize(vec4 q) {
    float len = length(q);
    return len > 1e-10 ? q / len : vec4(0.0, 0.0, 0.0, 1.0);
}

// Quaternion spherical linear interpolation (slerp)
vec4 quat_slerp(vec4 q0, vec4 q1, float t) {
    // Ensure shortest path
    float d = quat_dot(q0, q1);
    if (d < 0.0) {
        q1 = -q1;
        d = -d;
    }

    // If quaternions are very close, use linear interpolation
    if (d > 0.9995) {
        return quat_normalize(mix(q0, q1, t));
    }

    // Spherical interpolation
    float theta_0 = acos(clamp(d, -1.0, 1.0));
    float theta = theta_0 * t;
    float sin_theta = sin(theta);
    float sin_theta_0 = sin(theta_0);

    float s0 = cos(theta) - d * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;

    return quat_normalize(s0 * q0 + s1 * q1);
}

// Convert quaternion (x, y, z, w) to 3x3 rotation matrix
// GLSL mat3 is column-major: mat3(col0, col1, col2)
mat3 quat_to_matrix(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;

    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    // Column 0: (R00, R10, R20), Column 1: (R01, R11, R21), Column 2: (R02, R12, R22)
    return mat3(
        1.0 - (yy + zz), xy + wz, xz - wy,   // Column 0
        xy - wz, 1.0 - (xx + zz), yz + wx,   // Column 1
        xz + wy, yz - wx, 1.0 - (xx + yy)    // Column 2
    );
}

// Build 4x4 transform from translation and quaternion (with scale)
mat4 build_transform(vec3 translation, vec4 quaternion, vec3 scale) {
    mat3 rot = quat_to_matrix(quaternion);

    // Apply scale to each column of rotation matrix
    mat4 result = mat4(
        vec4(rot[0] * scale.x, 0.0),
        vec4(rot[1] * scale.y, 0.0),
        vec4(rot[2] * scale.z, 0.0),
        vec4(translation, 1.0)
    );

    return result;
}

// Binary search for track interpolation (returns index where t0 <= target < t1)
// Returns -1 if target is outside the track range
int binary_search_track(uint base_offset, uint count, int64_t target) {
    if (count < 2u) return -1;  // Need at least 2 points for interpolation

    int64_t first_ts = g_timestamps[base_offset];
    int64_t last_ts = g_timestamps[base_offset + count - 1u];

    // Check bounds
    if (target < first_ts || target > last_ts) return -1;

    int left = 0;
    int right = int(count) - 2;  // Max valid index for interpolation start
    int result = 0;

    while (left <= right) {
        int mid = (left + right) / 2;
        int64_t t0 = g_timestamps[base_offset + uint(mid)];
        int64_t t1 = g_timestamps[base_offset + uint(mid) + 1u];

        if (t0 <= target && target <= t1) {
            return mid;
        } else if (target < t0) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return -1;  // Should not reach here for valid input
}

//=============================================================================
// Spatial Culling Helpers
//=============================================================================

// AABB-AABB overlap test
bool cull_aabb_overlap(vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max) {
    return all(lessThanEqual(a_min, b_max)) && all(greaterThanEqual(a_max, b_min));
}

// Sphere-AABB overlap test
bool cull_sphere_aabb_overlap(vec3 center, float radius, vec3 b_min, vec3 b_max) {
    vec3 nearest = clamp(center, b_min, b_max);
    vec3 d = center - nearest;
    return dot(d, d) <= radius * radius;
}

// Extract camera world position from view pose
vec3 get_camera_world_pos(CameraPose pose) {
    mat3 R = mat3(pose.world_to_camera);
    return -transpose(R) * pose.world_to_camera[3].xyz;
}

//=============================================================================
// F-theta Projection
//=============================================================================

vec3 rotate_rodrigues(vec3 v, vec3 r) {
    float theta = length(r);
    if (theta < 1e-8) return v;
    vec3 k = r / theta;
    float c = cos(theta);
    float s = sin(theta);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

// F-theta projection: world point -> NDC
// Supports >180° FOV fisheye cameras where points can have negative z (behind camera plane)
vec4 ftheta_project(vec3 world_pos, CameraPose pose, FThetaCamera cam) {
    vec3 cam_pt = (pose.world_to_camera * vec4(world_pos, 1.0)).xyz;
    float depth = cam_pt.z;

    float ray_norm = length(cam_pt);

    // Handle points at camera origin
    if (ray_norm < 1e-6) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float half_pi = 1.5707963;  // π/2

    // For cameras with FOV <= 180° (max_ray_angle <= π/2), points behind camera should be clipped
    // Use pseudo-pinhole projection to push them far away in the correct direction
    if (cam.max_ray_angle <= half_pi && depth < 0.001) {
        float pseudo_focal = cam.poly1;
        float x_clip = cam_pt.x * pseudo_focal / (cam.img_w * 0.5);
        float y_clip = -cam_pt.y * pseudo_focal / (cam.img_h * 0.5);
        return vec4(x_clip * 10.0, y_clip * 10.0, 1.0, 1.0);  // Scale by 10 to push far outside
    }

    float xy_norm = length(cam_pt.xy);
    float cos_alpha = clamp(cam_pt.z / ray_norm, -1.0, 1.0);
    float alpha = acos(cos_alpha);  // alpha in [0, π]

    // Apply polynomial projection (with linear extrapolation beyond max_ray_angle)
    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a2 * a2;
    float a5 = a4 * alpha;

    float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                  cam.poly3 * a3 + cam.poly4 * a4 + cam.poly5 * a5;
    if (alpha > cam.max_ray_angle) {
        delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
    }

    float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
    vec2 pixel_rel = scale * cam_pt.xy;

    vec2 pixel_dist;
    pixel_dist.x = cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y;
    pixel_dist.y = cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y;

    vec2 pixel = pixel_dist + vec2(cam.cx, cam.cy);

    float x_ndc = 2.0 * pixel.x / cam.img_w - 1.0;
    float y_ndc = 1.0 - 2.0 * pixel.y / cam.img_h;

    // For z-buffer depth mapping:
    // - Narrow FOV (<=180°): use signed depth (cam_pt.z)
    // - Wide FOV (>180°): use ray_norm for front-camera vertices (unchanged),
    //   but push behind-camera vertices to depth_max so they never occlude
    //   forward geometry.  ray_norm alone can't distinguish front from behind.
    float z_value;
    if (cam.max_ray_angle > half_pi) {
        z_value = (depth >= 0.0) ? ray_norm : cam.depth_max;
    } else {
        z_value = depth;
    }
    float z_ndc = clamp(z_value / cam.depth_max, 0.0, 1.0);

    return vec4(x_ndc, y_ndc, z_ndc, 1.0);
}

// Inline version that takes mat4 directly
float estimate_edge_distortion_pixels_mat4(vec3 v0, vec3 v1, mat4 world_to_cam, FThetaCamera cam) {
    // Transform to camera space
    vec3 cam_pt0 = (world_to_cam * vec4(v0, 1.0)).xyz;
    vec3 cam_pt1 = (world_to_cam * vec4(v1, 1.0)).xyz;
    vec3 mid_world = (v0 + v1) * 0.5;
    vec3 cam_pt_mid = (world_to_cam * vec4(mid_world, 1.0)).xyz;

    // Just clamp depths to avoid division issues

    // For segments in front of camera, compute f-theta projection error directly
    // Use depth-clamped projection to handle near-plane cases
    float depth0 = max(cam_pt0.z, 0.001);
    float depth1 = max(cam_pt1.z, 0.001);
    float depth_mid = max(cam_pt_mid.z, 0.001);

    // Compute f-theta projection for each point
    // We inline the projection to avoid struct-passing issues
    vec2 pixel0, pixel1, pixel_mid;
    {
        float xy_norm = length(cam_pt0.xy);
        float ray_norm = length(vec3(cam_pt0.xy, depth0)) + 1e-10;
        float cos_alpha = clamp(depth0 / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt0.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel0 = pixel_dist + vec2(cam.cx, cam.cy);
    }
    {
        float xy_norm = length(cam_pt1.xy);
        float ray_norm = length(vec3(cam_pt1.xy, depth1)) + 1e-10;
        float cos_alpha = clamp(depth1 / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt1.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel1 = pixel_dist + vec2(cam.cx, cam.cy);
    }
    {
        float xy_norm = length(cam_pt_mid.xy);
        float ray_norm = length(vec3(cam_pt_mid.xy, depth_mid)) + 1e-10;
        float cos_alpha = clamp(depth_mid / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt_mid.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel_mid = pixel_dist + vec2(cam.cx, cam.cy);
    }

    // Compute error: distance from projected midpoint to linear interpolation
    vec2 linear_pixel_mid = (pixel0 + pixel1) * 0.5;
    float error_pixels = length(pixel_mid - linear_pixel_mid);

    // Clamp to reasonable range to handle edge cases
    return clamp(error_pixels, 0.0, 10000.0);
}

uint compute_subdivision_level(vec3 v0, vec3 v1, CameraPose pose, FThetaCamera cam, float threshold_pixels) {
    float error = estimate_edge_distortion_pixels_mat4(v0, v1, pose.world_to_camera, cam);

    if (error < threshold_pixels) return 0u;
    if (error < threshold_pixels * 2.0) return 1u;
    if (error < threshold_pixels * 4.0) return 2u;
    return 3u;
}

//=============================================================================
// Barycentric Subdivision Utilities (shared with polygon/cube)
//=============================================================================

uint bary_vertex_count(uint level) {
    // Vertices = (n+1)(n+2)/2 where n = 2^level
    if (level == 0u) return 3u;
    if (level == 1u) return 6u;
    if (level == 2u) return 15u;
    return 45u;  // level 3
}

uint bary_triangle_count(uint level) {
    // Triangles = 4^level
    if (level == 0u) return 1u;
    if (level == 1u) return 4u;
    if (level == 2u) return 16u;
    return 64u;  // level 3
}

vec2 bary_vertex_uv(uint vertex_idx, uint level) {
    if (level == 0u) {
        if (vertex_idx == 0u) return vec2(0.0, 0.0);
        if (vertex_idx == 1u) return vec2(1.0, 0.0);
        return vec2(0.0, 1.0);
    } else if (level == 1u) {
        vec2 uvs[6];
        uvs[0] = vec2(0.0, 0.0);
        uvs[1] = vec2(1.0, 0.0);
        uvs[2] = vec2(0.0, 1.0);
        uvs[3] = vec2(0.5, 0.0);
        uvs[4] = vec2(0.5, 0.5);
        uvs[5] = vec2(0.0, 0.5);
        return uvs[vertex_idx];
    } else if (level == 2u) {
        uint row, col;
        if (vertex_idx < 5u) { row = 0u; col = vertex_idx; }
        else if (vertex_idx < 9u) { row = 1u; col = vertex_idx - 5u; }
        else if (vertex_idx < 12u) { row = 2u; col = vertex_idx - 9u; }
        else if (vertex_idx < 14u) { row = 3u; col = vertex_idx - 12u; }
        else { row = 4u; col = 0u; }
        return vec2(float(col) / 4.0, float(row) / 4.0);
    } else {
        // Level 3: 9 rows (0-8), row r has (9-r) vertices
        uint row, col;
        if (vertex_idx < 9u) { row = 0u; col = vertex_idx; }
        else if (vertex_idx < 17u) { row = 1u; col = vertex_idx - 9u; }
        else if (vertex_idx < 24u) { row = 2u; col = vertex_idx - 17u; }
        else if (vertex_idx < 30u) { row = 3u; col = vertex_idx - 24u; }
        else if (vertex_idx < 35u) { row = 4u; col = vertex_idx - 30u; }
        else if (vertex_idx < 39u) { row = 5u; col = vertex_idx - 35u; }
        else if (vertex_idx < 42u) { row = 6u; col = vertex_idx - 39u; }
        else if (vertex_idx < 44u) { row = 7u; col = vertex_idx - 42u; }
        else { row = 8u; col = 0u; }
        return vec2(float(col) / 8.0, float(row) / 8.0);
    }
}

vec3 bary_interpolate(vec3 v0, vec3 v1, vec3 v2, vec2 uv) {
    float w = 1.0 - uv.x - uv.y;
    return v0 * w + v1 * uv.x + v2 * uv.y;
}

uvec3 bary_triangle_indices(uint tri_idx, uint level) {
    if (level == 0u) {
        return uvec3(0u, 1u, 2u);
    } else if (level == 1u) {
        uvec3 tris[4];
        tris[0] = uvec3(0u, 3u, 5u);
        tris[1] = uvec3(3u, 1u, 4u);
        tris[2] = uvec3(5u, 4u, 2u);
        tris[3] = uvec3(3u, 4u, 5u);
        return tris[tri_idx];
    } else if (level == 2u) {
        uint row_start[5] = uint[5](0u, 5u, 9u, 12u, 14u);
        uint t = tri_idx;
        uint row = 0u, col = 0u;
        bool is_down = false;

        if (t < 7u) {
            row = 0u;
            if (t < 4u) { col = t; is_down = false; }
            else { col = t - 4u; is_down = true; }
        } else if (t < 12u) {
            row = 1u;
            uint lt = t - 7u;
            if (lt < 3u) { col = lt; is_down = false; }
            else { col = lt - 3u; is_down = true; }
        } else if (t < 15u) {
            row = 2u;
            uint lt = t - 12u;
            if (lt < 2u) { col = lt; is_down = false; }
            else { col = lt - 2u; is_down = true; }
        } else {
            row = 3u; col = 0u; is_down = false;
        }

        uint i0, i1, i2;
        if (!is_down) {
            i0 = row_start[row] + col;
            i1 = row_start[row] + col + 1u;
            i2 = row_start[row + 1u] + col;
        } else {
            i0 = row_start[row] + col + 1u;
            i1 = row_start[row + 1u] + col + 1u;
            i2 = row_start[row + 1u] + col;
        }
        return uvec3(i0, i1, i2);
    } else {
        // Level 3: row_start for 9 rows
        uint row_start[9] = uint[9](0u, 9u, 17u, 24u, 30u, 35u, 39u, 42u, 44u);
        uint t = tri_idx;
        uint row = 0u, col = 0u;
        bool is_down = false;

        // Row 0: 8 up + 7 down = 15, Row 1: 7 up + 6 down = 13, etc.
        if (t < 15u) {
            row = 0u;
            if (t < 8u) { col = t; is_down = false; }
            else { col = t - 8u; is_down = true; }
        } else if (t < 28u) {
            row = 1u; uint lt = t - 15u;
            if (lt < 7u) { col = lt; is_down = false; }
            else { col = lt - 7u; is_down = true; }
        } else if (t < 39u) {
            row = 2u; uint lt = t - 28u;
            if (lt < 6u) { col = lt; is_down = false; }
            else { col = lt - 6u; is_down = true; }
        } else if (t < 48u) {
            row = 3u; uint lt = t - 39u;
            if (lt < 5u) { col = lt; is_down = false; }
            else { col = lt - 5u; is_down = true; }
        } else if (t < 55u) {
            row = 4u; uint lt = t - 48u;
            if (lt < 4u) { col = lt; is_down = false; }
            else { col = lt - 4u; is_down = true; }
        } else if (t < 60u) {
            row = 5u; uint lt = t - 55u;
            if (lt < 3u) { col = lt; is_down = false; }
            else { col = lt - 3u; is_down = true; }
        } else if (t < 63u) {
            row = 6u; uint lt = t - 60u;
            if (lt < 2u) { col = lt; is_down = false; }
            else { col = lt - 2u; is_down = true; }
        } else {
            row = 7u; col = 0u; is_down = false;
        }

        uint i0, i1, i2;
        if (!is_down) {
            i0 = row_start[row] + col;
            i1 = row_start[row] + col + 1u;
            i2 = row_start[row + 1u] + col;
        } else {
            i0 = row_start[row] + col + 1u;
            i1 = row_start[row + 1u] + col + 1u;
            i2 = row_start[row + 1u] + col;
        }
        return uvec3(i0, i1, i2);
    }
}


layout(local_size_x = 32) in;
// No per-vertex output arrays - all gradient data passed per-primitive
// This allows unlimited vertices (within mesh shader spec)
layout(triangles, max_vertices = 96, max_primitives = 128) out;

struct ObstacleTaskPayload {
    uint query_id;
    uint obstacle_id;
    mat4 object_to_world;
    vec3 scale;
    vec3 front_color;
    vec3 back_color;
    uint subdivision_level;  // From task shader
    uint render_flags;       // CUBE_FLAG_* bits from pool
    float is_bev;
    uint face_mask;          // Bitmask of front-facing faces
};
taskPayloadSharedEXT ObstacleTaskPayload payload;

out gl_MeshPerVertexNV {
    vec4 gl_Position;
} gl_MeshVerticesEXT[];

perprimitiveEXT out gl_MeshPerPrimitiveNV {
    int gl_Layer;
} gl_MeshPrimitivesEXT[];

// Per-primitive gradient data (no per-vertex outputs needed!)
layout(location = 0) perprimitiveEXT out CubeGradientBlock {
    vec3 corner_t;      // gradient t values at triangle's 3 corners
    vec3 front_color;
    vec3 back_color;
    float is_bev;       // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_out[];

// Unit cube vertices
const vec3 CUBE_VERTS[8] = vec3[8](
    vec3(-0.5, -0.5, -0.5), vec3(+0.5, -0.5, -0.5),
    vec3(+0.5, +0.5, -0.5), vec3(-0.5, +0.5, -0.5),
    vec3(-0.5, -0.5, +0.5), vec3(+0.5, -0.5, +0.5),
    vec3(+0.5, +0.5, +0.5), vec3(-0.5, +0.5, +0.5)
);

// Face vertex indices (CCW winding)
const uvec4 FACE_VERTS[6] = uvec4[6](
    uvec4(0, 3, 2, 1),  // -Z (back)
    uvec4(4, 5, 6, 7),  // +Z (front)
    uvec4(0, 4, 7, 3),  // -X
    uvec4(1, 2, 6, 5),  // +X
    uvec4(0, 1, 5, 4),  // -Y
    uvec4(3, 7, 6, 2)   // +Y
);

// 12 edges of cube (pairs of vertex indices)
const uvec2 CUBE_EDGES[12] = uvec2[12](
    uvec2(0,1), uvec2(1,2), uvec2(2,3), uvec2(3,0),  // back face
    uvec2(4,5), uvec2(5,6), uvec2(6,7), uvec2(7,4),  // front face
    uvec2(0,4), uvec2(1,5), uvec2(2,6), uvec2(3,7)   // connecting edges
);

// Edge wireframe color (light gray - matches reference: [200, 200, 200] / 255.0)
const vec3 EDGE_COLOR = vec3(0.784, 0.784, 0.784);

void main() {
    uint _prim_count = 0u;
    uint workgroup_id = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    RenderQuery query = g_queries[payload.query_id];
    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose view_pose = g_camera_poses[payload.query_id];

    // Workgroups 6-11 = wireframe edges (2 edges per workgroup, subdivided to match face tessellation)
    if (workgroup_id >= 6u) {
        uint wg_edge_offset = (workgroup_id - 6u) * 2u;  // 0, 2, 4, 6, 8, 10
        float EDGE_WIDTH = get_wireframe_width();

        // Edge-to-face adjacency for backface culling of wireframe edges
        const uvec2 EDGE_FACES[12] = uvec2[12](
            uvec2(0, 4), uvec2(0, 3), uvec2(0, 5), uvec2(0, 2),
            uvec2(1, 4), uvec2(1, 3), uvec2(1, 5), uvec2(1, 2),
            uvec2(2, 4), uvec2(3, 4), uvec2(3, 5), uvec2(2, 5)
        );

        // Check if either edge in this workgroup is adjacent to a visible face
        uint e0 = wg_edge_offset;
        uint e1 = wg_edge_offset + 1u;
        bool e0_vis = (payload.face_mask & ((1u << EDGE_FACES[e0].x) | (1u << EDGE_FACES[e0].y))) != 0u;
        bool e1_vis = (payload.face_mask & ((1u << EDGE_FACES[e1].x) | (1u << EDGE_FACES[e1].y))) != 0u;

        if (!e0_vis && !e1_vis) {
            _prim_count = 0u; SetMeshOutputsEXT(0u, _prim_count);
            return;
        }

        // Number of segments per edge based on subdivision level (matches face tessellation)
        // subdiv 0 = 1 segment, subdiv 1 = 2 segments, subdiv 2 = 4 segments
        uint segs_per_edge = 1u << payload.subdivision_level;  // 1, 2, or 4
        uint total_segments = segs_per_edge * 2u;  // 2 edges

        uint total_edge_verts = total_segments * 4u;
        _prim_count = total_segments * 2u; SetMeshOutputsEXT(total_edge_verts, _prim_count);  // 2 triangles per segment quad

        // Process both edges in this workgroup
        for (uint local_seg = tid; local_seg < total_segments; local_seg += 32u) {
            uint edge_in_wg = local_seg / segs_per_edge;  // 0 or 1
            uint seg_in_edge = local_seg % segs_per_edge;
            uint edge_id = wg_edge_offset + edge_in_wg;

            // Skip edges adjacent only to back-facing faces
            uvec2 adj = EDGE_FACES[edge_id];
            if ((payload.face_mask & ((1u << adj.x) | (1u << adj.y))) == 0u) {
                uint base_vert = local_seg * 4u;
                gl_MeshVerticesEXT[base_vert].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesEXT[base_vert + 1u].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesEXT[base_vert + 2u].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesEXT[base_vert + 3u].gl_Position = vec4(0, 0, 0, 1);
                continue;
            }

            // Get edge endpoints
            uvec2 edge = CUBE_EDGES[edge_id];
            vec3 v0_local = CUBE_VERTS[edge.x] * payload.scale;
            vec3 v1_local = CUBE_VERTS[edge.y] * payload.scale;

            vec3 v0_world = (payload.object_to_world * vec4(v0_local, 1.0)).xyz;
            vec3 v1_world = (payload.object_to_world * vec4(v1_local, 1.0)).xyz;

            // Interpolate two world-space points for this segment
            float t0 = float(seg_in_edge) / float(segs_per_edge);
            float t1 = float(seg_in_edge + 1u) / float(segs_per_edge);

            vec3 pt0_world = mix(v0_world, v1_world, t0);
            vec3 pt1_world = mix(v0_world, v1_world, t1);

            // Project to clip space (f-theta curvature applied here)
            vec4 clip0 = ftheta_project(pt0_world, view_pose, cam);
            vec4 clip1 = ftheta_project(pt1_world, view_pose, cam);

            // Edge direction and perpendicular for width
            vec2 ndc0 = clip0.xy / clip0.w;
            vec2 ndc1 = clip1.xy / clip1.w;
            vec2 edge_dir = normalize(ndc1 - ndc0);
            vec2 perp = vec2(-edge_dir.y, edge_dir.x);
            vec2 offset = perp * EDGE_WIDTH / vec2(cam.img_w, cam.img_h);

            // Create 4 vertices for the quad
            uint base_vert = local_seg * 4u;

            vec4 off0 = vec4(offset * clip0.w, 0.0, 0.0);
            vec4 off1 = vec4(offset * clip1.w, 0.0, 0.0);

            // Depth bias to render edges on top of faces
            float z_bias0 = -0.001 * clip0.w;
            float z_bias1 = -0.001 * clip1.w;

            vec4 p0 = clip0 - off0; p0.z += z_bias0;
            vec4 p1 = clip0 + off0; p1.z += z_bias0;
            vec4 p2 = clip1 + off1; p2.z += z_bias1;
            vec4 p3 = clip1 - off1; p3.z += z_bias1;

            gl_MeshVerticesEXT[base_vert + 0u].gl_Position = p0;
            gl_MeshVerticesEXT[base_vert + 1u].gl_Position = p1;
            gl_MeshVerticesEXT[base_vert + 2u].gl_Position = p2;
            gl_MeshVerticesEXT[base_vert + 3u].gl_Position = p3;
        }

        barrier();

        // Generate triangle indices for segment quads
        for (uint local_seg = tid; local_seg < total_segments; local_seg += 32u) {
            uint base_vert = local_seg * 4u;
            uint base_tri = local_seg * 2u;

            // Triangle 1: 0, 1, 2
            gl_PrimitiveTriangleIndicesEXT[base_tri + 0u] = uvec3(base_vert + 0u, base_vert + 1u, base_vert + 2u);
            gl_MeshPrimitivesEXT[base_tri + 0u].gl_Layer = int(payload.query_id);
            // Wireframe uses solid edge color (front = back = EDGE_COLOR, corner_t doesn't matter)
            prim_out[base_tri + 0u].corner_t = vec3(0.5);
            prim_out[base_tri + 0u].front_color = EDGE_COLOR;
            prim_out[base_tri + 0u].back_color = EDGE_COLOR;
            prim_out[base_tri + 0u].is_bev = payload.is_bev;

            // Triangle 2: 0, 2, 3
            gl_PrimitiveTriangleIndicesEXT[base_tri + 1u] = uvec3(base_vert + 0u, base_vert + 2u, base_vert + 3u);
            gl_MeshPrimitivesEXT[base_tri + 1u].gl_Layer = int(payload.query_id);
            prim_out[base_tri + 1u].corner_t = vec3(0.5);
            prim_out[base_tri + 1u].front_color = EDGE_COLOR;
            prim_out[base_tri + 1u].back_color = EDGE_COLOR;
            prim_out[base_tri + 1u].is_bev = payload.is_bev;
        }

        return;
    }

    // Workgroups 0-5: Face rendering (2 triangles per face)
    uint face_id = workgroup_id;
    uint subdiv = payload.subdivision_level;

    // Backface culling: skip faces not visible from camera
    if ((payload.face_mask & (1u << face_id)) == 0u) {
        _prim_count = 0u; SetMeshOutputsEXT(0u, _prim_count);
        return;
    }

    // Get face corner info
    uvec4 face = FACE_VERTS[face_id];

    // Compute per-corner gradient t values and world positions
    float corner_t[4];
    vec3 corners[4];
    for (uint i = 0u; i < 4u; i++) {
        uint vert_idx = (i == 0u) ? face.x : (i == 1u) ? face.y : (i == 2u) ? face.z : face.w;
        vec3 local_vert = CUBE_VERTS[vert_idx];
        corner_t[i] = local_vert.x + 0.5;  // 0 at back (-X), 1 at front (+X)
        corners[i] = (payload.object_to_world * vec4(local_vert * payload.scale, 1.0)).xyz;
    }

    // Use shared barycentric subdivision utilities
    uint verts_per_tri = bary_vertex_count(subdiv);
    uint tris_per_tri = bary_triangle_count(subdiv);

    // 2 triangles per face
    uint total_verts = verts_per_tri * 2u;
    uint total_tris = tris_per_tri * 2u;

    _prim_count = total_tris; SetMeshOutputsEXT(total_verts, _prim_count);

    // Face = 2 triangles: (c0,c1,c2) and (c0,c2,c3)
    vec3 tri_verts[6];
    float tri_t[6];
    tri_verts[0] = corners[0]; tri_verts[1] = corners[1]; tri_verts[2] = corners[2];
    tri_verts[3] = corners[0]; tri_verts[4] = corners[2]; tri_verts[5] = corners[3];
    tri_t[0] = corner_t[0]; tri_t[1] = corner_t[1]; tri_t[2] = corner_t[2];
    tri_t[3] = corner_t[0]; tri_t[4] = corner_t[2]; tri_t[5] = corner_t[3];

    // Generate vertices for both triangles (position only, no per-vertex outputs)
    for (uint t = 0u; t < 2u; t++) {
        vec3 v0 = tri_verts[t * 3u];
        vec3 v1 = tri_verts[t * 3u + 1u];
        vec3 v2 = tri_verts[t * 3u + 2u];
        uint vert_base = t * verts_per_tri;

        for (uint v = tid; v < verts_per_tri; v += 32u) {
            vec2 uv = bary_vertex_uv(v, subdiv);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);
            vec4 clip = ftheta_project(pt, view_pose, cam);
            gl_MeshVerticesEXT[vert_base + v].gl_Position = clip;
        }
    }

    barrier();

    // Generate triangle indices and set per-primitive data (including corner_t)
    for (uint t = 0u; t < 2u; t++) {
        uint vert_base = t * verts_per_tri;
        uint tri_base = t * tris_per_tri;
        float t0 = tri_t[t * 3u];
        float t1 = tri_t[t * 3u + 1u];
        float t2 = tri_t[t * 3u + 2u];

        for (uint i = tid; i < tris_per_tri; i += 32u) {
            uvec3 idx = bary_triangle_indices(i, subdiv);
            uint idx_base = (tri_base + i) * 3u;
            gl_PrimitiveTriangleIndicesEXT[(idx_base) / 3u] = uvec3(vert_base + idx.x, vert_base + idx.y, vert_base + idx.z);
            gl_MeshPrimitivesEXT[tri_base + i].gl_Layer = int(payload.query_id);

            // Compute gradient t values at each corner of this sub-triangle
            vec2 uv0 = bary_vertex_uv(idx.x, subdiv);
            vec2 uv1 = bary_vertex_uv(idx.y, subdiv);
            vec2 uv2 = bary_vertex_uv(idx.z, subdiv);
            vec3 corner_t_values = vec3(
                (1.0 - uv0.x - uv0.y) * t0 + uv0.x * t1 + uv0.y * t2,
                (1.0 - uv1.x - uv1.y) * t0 + uv1.x * t1 + uv1.y * t2,
                (1.0 - uv2.x - uv2.y) * t0 + uv2.x * t1 + uv2.y * t2
            );

            // Set per-primitive gradient data
            prim_out[tri_base + i].corner_t = corner_t_values;
            prim_out[tri_base + i].front_color = payload.front_color;
            prim_out[tri_base + i].back_color = payload.back_color;
            prim_out[tri_base + i].is_bev = payload.is_bev;
        }
    }
}
