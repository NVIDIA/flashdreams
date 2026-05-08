import "slangpy";

struct LinePrimitive
{
    float4 p0;
    float4 p1;
    float4 color;
    float4 params;
};

struct TrianglePrimitive
{
    float4 p0;
    float4 p1;
    float4 p2;
    float4 color;
};

float point_triangle_distance_squared(float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
        return dot(ap, ap);

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
        return dot(bp, bp);

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = d1 / max(d1 - d3, 1e-6);
        float3 closest = a + v * ab;
        return dot(p - closest, p - closest);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
        return dot(cp, cp);

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float w = d2 / max(d2 - d6, 1e-6);
        float3 closest = a + w * ac;
        return dot(p - closest, p - closest);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float3 bc = c - b;
        float w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-6);
        float3 closest = b + w * bc;
        return dot(p - closest, p - closest);
    }

    float denom = 1.0 / max(va + vb + vc, 1e-6);
    float v = vb * denom;
    float w = vc * denom;
    float3 closest = a + ab * v + ac * w;
    return dot(p - closest, p - closest);
}

bool ray_triangle_intersect(
    float3 ray_dir,
    float3 v0,
    float3 v1,
    float3 v2,
    out float t,
    out float3 bary)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ray_dir, e2);
    float det = dot(e1, pvec);
    if (abs(det) <= 1e-6)
    {
        t = 0.0;
        bary = 0.0;
        return false;
    }

    float inv_det = 1.0 / det;
    float3 tvec = -v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
    {
        t = 0.0;
        bary = 0.0;
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(ray_dir, qvec) * inv_det;
    if (v < 0.0 || (u + v) > 1.0)
    {
        t = 0.0;
        bary = 0.0;
        return false;
    }

    t = dot(e2, qvec) * inv_det;
    bary = float3(1.0 - u - v, u, v);
    return t > 0.0;
}

float point_segment_distance(float2 p, float2 a, float2 b, out float t)
{
    float2 ab = b - a;
    float denom = max(dot(ab, ab), 1e-6);
    t = saturate(dot(p - a, ab) / denom);
    float2 closest = a + t * ab;
    return length(p - closest);
}

bool barycentrics(float2 p, float2 a, float2 b, float2 c, out float3 w)
{
    float2 v0 = b - a;
    float2 v1 = c - a;
    float2 v2 = p - a;
    float denom = v0.x * v1.y - v1.x * v0.y;
    if (abs(denom) < 1e-6)
    {
        w = 0.0;
        return false;
    }

    float inv_denom = 1.0 / denom;
    float v = (v2.x * v1.y - v1.x * v2.y) * inv_denom;
    float u = (v0.x * v2.y - v2.x * v0.y) * inv_denom;
    float w0 = 1.0 - u - v;
    w = float3(w0, v, u);
    return all(w >= 0.0);
}

float angle_to_radius(
    float angle,
    StructuredBuffer<float> angle_to_radius_lut,
    uint lut_count,
    float lut_max_angle,
    float lut_max_radius,
    float lut_tail_slope)
{
    if (lut_count == 0)
        return 0.0;
    if (lut_count == 1)
        return angle_to_radius_lut[0];

    if (angle > lut_max_angle)
        return lut_max_radius + (angle - lut_max_angle) * lut_tail_slope;

    float scaled = clamp(angle / max(lut_max_angle, 1e-6), 0.0, 1.0) * float(lut_count - 1);
    uint i0 = min(uint(floor(scaled)), lut_count - 1);
    uint i1 = min(i0 + 1, lut_count - 1);
    float t = scaled - float(i0);
    return lerp(angle_to_radius_lut[i0], angle_to_radius_lut[i1], t);
}

float radius_to_angle(
    float radius,
    StructuredBuffer<float> radius_to_angle_lut,
    uint lut_count,
    float lut_max_radius,
    float lut_tail_slope)
{
    if (lut_count == 0)
        return 0.0;
    if (lut_count == 1)
        return radius_to_angle_lut[0];

    if (radius > lut_max_radius)
        return radius_to_angle_lut[lut_count - 1] + (radius - lut_max_radius) * lut_tail_slope;

    float scaled = clamp(radius / max(lut_max_radius, 1e-6), 0.0, 1.0) * float(lut_count - 1);
    uint i0 = min(uint(floor(scaled)), lut_count - 1);
    uint i1 = min(i0 + 1, lut_count - 1);
    float t = scaled - float(i0);
    return lerp(radius_to_angle_lut[i0], radius_to_angle_lut[i1], t);
}

float3 pixel_to_camera_ray(
    float2 pixel_center,
    float2 principal_px,
    float2 inv_linear_row0,
    float2 inv_linear_row1,
    StructuredBuffer<float> radius_to_angle_lut,
    uint lut_count,
    float lut_max_radius,
    float lut_tail_slope)
{
    float2 pixel_offset = pixel_center - principal_px;
    float2 pixels_rel = float2(dot(inv_linear_row0, pixel_offset), dot(inv_linear_row1, pixel_offset));
    float radius = length(pixels_rel);
    float angle = radius_to_angle(radius, radius_to_angle_lut, lut_count, lut_max_radius, lut_tail_slope);
    float2 dir_xy = radius > 1e-6 ? pixels_rel / radius : float2(0.0, 0.0);
    float sin_angle = sin(angle);
    return float3(dir_xy * sin_angle, cos(angle));
}

float2 project_to_polygon_plane(float3 point, uint dominant_axis)
{
    if (dominant_axis == 0)
        return point.yz;
    if (dominant_axis == 1)
        return point.xz;
    return point.xy;
}

bool point_in_polygon_camera(
    float3 point,
    StructuredBuffer<float4> polygon_vertices_camera,
    uint vertex_offset,
    uint vertex_count,
    uint dominant_axis)
{
    if (vertex_count < 3)
        return false;

    float2 point_2d = project_to_polygon_plane(point, dominant_axis);
    bool inside = false;
    float2 previous = project_to_polygon_plane(polygon_vertices_camera[vertex_offset + vertex_count - 1].xyz, dominant_axis);

    [loop]
    for (uint i = 0; i < vertex_count; ++i)
    {
        float2 current = project_to_polygon_plane(polygon_vertices_camera[vertex_offset + i].xyz, dominant_axis);
        float denom = previous.y - current.y;
        if (abs(denom) <= 1e-6)
            denom = denom >= 0.0 ? 1e-6 : -1e-6;
        bool crosses = ((current.y > point_2d.y) != (previous.y > point_2d.y)) &&
            (point_2d.x < (previous.x - current.x) * (point_2d.y - current.y) / denom + current.x);
        if (crosses)
            inside = !inside;
        previous = current;
    }

    return inside;
}

struct ProjectedPoint
{
    bool valid;
    float2 uv;
    float forwardDepth;
};

float3 world_to_camera_point(float3 world_point, float4x4 world_to_camera_rdf)
{
    return mul(world_to_camera_rdf, float4(world_point, 1.0)).xyz;
}

ProjectedPoint project_camera_point(
    float3 camera_point,
    float2 principal_px,
    float2 linear_row0,
    float2 linear_row1,
    float near_plane_m,
    StructuredBuffer<float> angle_to_radius_lut,
    uint lut_count,
    float lut_max_angle,
    float lut_max_radius,
    float lut_tail_slope)
{
    float forward_depth = camera_point.z;
    if (forward_depth <= near_plane_m)
    {
        return { false, float2(0.0, 0.0), 0.0 };
    }

    float xy_norm = length(camera_point.xy);
    float ray_norm = length(camera_point);
    float cos_alpha = clamp(forward_depth / max(ray_norm, 1e-6), -1.0, 1.0);
    float alpha = acos(cos_alpha);
    float radius = angle_to_radius(alpha, angle_to_radius_lut, lut_count, lut_max_angle, lut_max_radius, lut_tail_slope);

    float2 pixels_rel = xy_norm > 1e-6 ? camera_point.xy * (radius / xy_norm) : float2(0.0, 0.0);
    float2 uv = float2(dot(linear_row0, pixels_rel), dot(linear_row1, pixels_rel)) + principal_px;
    return { true, uv, forward_depth };
}

ProjectedPoint project_point(
    float3 world_point,
    float4x4 world_to_camera_rdf,
    float2 principal_px,
    float2 linear_row0,
    float2 linear_row1,
    float near_plane_m,
    StructuredBuffer<float> angle_to_radius_lut,
    uint lut_count,
    float lut_max_angle,
    float lut_max_radius,
    float lut_tail_slope)
{
    return project_camera_point(
        world_to_camera_point(world_point, world_to_camera_rdf),
        principal_px,
        linear_row0,
        linear_row1,
        near_plane_m,
        angle_to_radius_lut,
        lut_count,
        lut_max_angle,
        lut_max_radius,
        lut_tail_slope
    );
}

uint clip_triangle_to_near_plane(float3 v0, float3 v1, float3 v2, float near_plane_m, out float3 clipped[4])
{
    float clip_plane_z = near_plane_m + 1e-4;
    float3 input[3] = { v0, v1, v2 };
    uint out_count = 0;
    float3 previous = input[2];
    bool previous_inside = previous.z >= clip_plane_z;

    [unroll]
    for (uint i = 0; i < 3; ++i)
    {
        float3 current = input[i];
        bool current_inside = current.z >= clip_plane_z;

        if (previous_inside != current_inside)
        {
            float denom = current.z - previous.z;
            float t = abs(denom) > 1e-6 ? (clip_plane_z - previous.z) / denom : 0.0;
            clipped[out_count++] = lerp(previous, current, saturate(t));
        }

        if (current_inside)
            clipped[out_count++] = current;

        previous = current;
        previous_inside = current_inside;
    }

    return out_count;
}

float fog_amount(float depth_value, float fog_start_m, float fog_end_m, float fog_power)
{
    if (fog_end_m <= fog_start_m)
        return 0.0;
    float t = saturate((depth_value - fog_start_m) / max(fog_end_m - fog_start_m, 1e-6));
    return pow(t, max(fog_power, 1e-3));
}

float4 apply_distance_fog(float4 color, float depth_value, float fog_start_m, float fog_end_m, float fog_power)
{
    float amount = fog_amount(depth_value, fog_start_m, fog_end_m, fog_power);
    return float4(color.rgb * (1.0 - amount), color.a);
}

void rasterize_projected_triangle(
    float2 a,
    float za,
    float2 b,
    float zb,
    float2 c,
    float zc,
    float4 color,
    RWTensor<float4, 2> rgb,
    RWTensor<float, 2> depth,
    RWTensor<uint, 2> depth_bits,
    uint image_width,
    uint image_height,
    float fog_start_m,
    float fog_end_m,
    float fog_power)
{
    int x0 = max(0, int(floor(min(a.x, min(b.x, c.x)))));
    int y0 = max(0, int(floor(min(a.y, min(b.y, c.y)))));
    int x1 = min(int(image_width) - 1, int(ceil(max(a.x, max(b.x, c.x)))));
    int y1 = min(int(image_height) - 1, int(ceil(max(a.y, max(b.y, c.y)))));

    for (int py = y0; py <= y1; ++py)
    {
        for (int px = x0; px <= x1; ++px)
        {
            float2 pixel_center = float2(px + 0.5, py + 0.5);
            float3 weights;
            if (!barycentrics(pixel_center, a, b, c, weights))
                continue;

            float depth_value = weights.x * za + weights.y * zb + weights.z * zc;
            uint depth_value_bits = asuint(depth_value);
            uint old_bits;
            int linear_idx = py * int(depth_bits._strides[0]) + px * int(depth_bits._strides[1]) + int(depth_bits._offset);
            InterlockedMin(depth_bits._data[linear_idx], depth_value_bits, old_bits);
            if (depth_value_bits < old_bits)
            {
                depth.store(py, px, depth_value);
                rgb.store(py, px, apply_distance_fog(color, depth_value, fog_start_m, fog_end_m, fog_power));
            }
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clear_targets(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform float clear_depth)
{
    if (dispatchThreadID.x >= image_width || dispatchThreadID.y >= image_height)
        return;

    int px = int(dispatchThreadID.x);
    int py = int(dispatchThreadID.y);
    rgb.store(py, px, float4(0.0, 0.0, 0.0, 1.0));
    depth.store(py, px, clear_depth);
    depth_bits.store(py, px, asuint(clear_depth));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_lines_world(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<LinePrimitive> lines,
    uniform StructuredBuffer<float> angle_to_radius_lut,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform float4x4 world_to_camera_rdf,
    uniform float2 principal_px,
    uniform float2 linear_row0,
    uniform float2 linear_row1,
    uniform float near_plane_m,
    uniform uint lut_count,
    uniform float lut_max_angle,
    uniform float lut_max_radius,
    uniform float lut_tail_slope,
    uniform float fog_start_m,
    uniform float fog_end_m,
    uniform float fog_power)
{
    uint idx = dispatchThreadID.x;
    uint line_count;
    uint line_stride;
    lines.GetDimensions(line_count, line_stride);
    if (idx >= line_count)
        return;

    LinePrimitive line = lines[idx];
    ProjectedPoint p0 = project_point(
        line.p0.xyz,
        world_to_camera_rdf,
        principal_px,
        linear_row0,
        linear_row1,
        near_plane_m,
        angle_to_radius_lut,
        lut_count,
        lut_max_angle,
        lut_max_radius,
        lut_tail_slope
    );
    ProjectedPoint p1 = project_point(
        line.p1.xyz,
        world_to_camera_rdf,
        principal_px,
        linear_row0,
        linear_row1,
        near_plane_m,
        angle_to_radius_lut,
        lut_count,
        lut_max_angle,
        lut_max_radius,
        lut_tail_slope
    );
    if (!p0.valid || !p1.valid)
        return;

    float2 a = p0.uv;
    float2 b = p1.uv;
    float za = p0.forwardDepth;
    float zb = p1.forwardDepth;
    float4 color = line.color;
    float half_width = max(line.params.x * 0.5, 0.5);

    int x0 = max(0, int(floor(min(a.x, b.x) - half_width)));
    int y0 = max(0, int(floor(min(a.y, b.y) - half_width)));
    int x1 = min(int(image_width) - 1, int(ceil(max(a.x, b.x) + half_width)));
    int y1 = min(int(image_height) - 1, int(ceil(max(a.y, b.y) + half_width)));

    for (int py = y0; py <= y1; ++py)
    {
        for (int px = x0; px <= x1; ++px)
        {
            float2 pixel_center = float2(px + 0.5, py + 0.5);
            float t;
            float distance_to_line = point_segment_distance(pixel_center, a, b, t);
            if (distance_to_line > half_width)
                continue;

            float depth_value = lerp(za, zb, t);
            uint depth_value_bits = asuint(depth_value);
            uint old_bits;
            int linear_idx = py * int(depth_bits._strides[0]) + px * int(depth_bits._strides[1]) + int(depth_bits._offset);
            InterlockedMin(depth_bits._data[linear_idx], depth_value_bits, old_bits);
            if (depth_value_bits < old_bits)
            {
                depth.store(py, px, depth_value);
                rgb.store(py, px, apply_distance_fog(color, depth_value, fog_start_m, fog_end_m, fog_power));
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_triangles_world(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<TrianglePrimitive> triangles,
    uniform StructuredBuffer<uint> near_triangle_mask,
    uniform StructuredBuffer<float> angle_to_radius_lut,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform float4x4 world_to_camera_rdf,
    uniform float2 principal_px,
    uniform float2 linear_row0,
    uniform float2 linear_row1,
    uniform float near_plane_m,
    uniform uint lut_count,
    uniform float lut_max_angle,
    uniform float lut_max_radius,
    uniform float lut_tail_slope,
    uniform uint skip_masked,
    uniform float fog_start_m,
    uniform float fog_end_m,
    uniform float fog_power)
{
    uint idx = dispatchThreadID.x;
    uint triangle_count;
    uint triangle_stride;
    triangles.GetDimensions(triangle_count, triangle_stride);
    if (idx >= triangle_count)
        return;
    if (skip_masked != 0 && near_triangle_mask[idx] != 0)
        return;

    TrianglePrimitive tri = triangles[idx];
    float3 clipped_camera_points[4];
    uint clipped_count = clip_triangle_to_near_plane(
        world_to_camera_point(tri.p0.xyz, world_to_camera_rdf),
        world_to_camera_point(tri.p1.xyz, world_to_camera_rdf),
        world_to_camera_point(tri.p2.xyz, world_to_camera_rdf),
        near_plane_m,
        clipped_camera_points
    );
    if (clipped_count < 3)
        return;

    ProjectedPoint projected[4];
    [unroll]
    for (uint i = 0; i < clipped_count; ++i)
    {
        projected[i] = project_camera_point(
            clipped_camera_points[i],
            principal_px,
            linear_row0,
            linear_row1,
            near_plane_m,
            angle_to_radius_lut,
            lut_count,
            lut_max_angle,
            lut_max_radius,
            lut_tail_slope
        );
    }

    float4 color = tri.color;
    rasterize_projected_triangle(
        projected[0].uv,
        projected[0].forwardDepth,
        projected[1].uv,
        projected[1].forwardDepth,
        projected[2].uv,
        projected[2].forwardDepth,
        color,
        rgb,
        depth,
        depth_bits,
        image_width,
        image_height,
        fog_start_m,
        fog_end_m,
        fog_power
    );

    if (clipped_count == 4)
    {
        rasterize_projected_triangle(
            projected[0].uv,
            projected[0].forwardDepth,
            projected[2].uv,
            projected[2].forwardDepth,
            projected[3].uv,
            projected[3].forwardDepth,
            color,
            rgb,
            depth,
            depth_bits,
            image_width,
            image_height,
            fog_start_m,
            fog_end_m,
            fog_power
        );
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify_near_triangles_world(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<TrianglePrimitive> triangles,
    uniform RWStructuredBuffer<uint> near_triangle_mask,
    uniform RWStructuredBuffer<uint> near_triangle_indices,
    uniform RWStructuredBuffer<uint> counters,
    uniform float4x4 world_to_camera_rdf,
    uniform float triangle_raytrace_distance_m)
{
    uint idx = dispatchThreadID.x;
    uint triangle_count;
    uint triangle_stride;
    triangles.GetDimensions(triangle_count, triangle_stride);
    if (idx >= triangle_count)
        return;

    TrianglePrimitive tri = triangles[idx];
    float3 v0 = world_to_camera_point(tri.p0.xyz, world_to_camera_rdf);
    float3 v1 = world_to_camera_point(tri.p1.xyz, world_to_camera_rdf);
    float3 v2 = world_to_camera_point(tri.p2.xyz, world_to_camera_rdf);
    float max_depth = max(v0.z, max(v1.z, v2.z));
    if (max_depth <= 0.0)
        return;

    float threshold_sq = triangle_raytrace_distance_m * triangle_raytrace_distance_m;
    float distance_sq = point_triangle_distance_squared(float3(0.0, 0.0, 0.0), v0, v1, v2);
    if (distance_sq >= threshold_sq)
        return;

    near_triangle_mask[idx] = 1;
    uint near_slot;
    InterlockedAdd(counters[0], 1, near_slot);
    uint index_capacity;
    uint index_stride;
    near_triangle_indices.GetDimensions(index_capacity, index_stride);
    if (near_slot < index_capacity)
        near_triangle_indices[near_slot] = idx;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void raytrace_triangles_world(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> near_triangle_indices,
    uniform StructuredBuffer<TrianglePrimitive> triangles,
    uniform StructuredBuffer<float> radius_to_angle_lut,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform float4x4 world_to_camera_rdf,
    uniform float2 principal_px,
    uniform float2 inv_linear_row0,
    uniform float2 inv_linear_row1,
    uniform float near_plane_m,
    uniform uint radius_lut_count,
    uniform float radius_lut_max_radius,
    uniform float radius_lut_tail_slope,
    uniform uint near_triangle_count,
    uniform float fog_start_m,
    uniform float fog_end_m,
    uniform float fog_power)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= image_width || py >= image_height || near_triangle_count == 0)
        return;

    float2 pixel_center = float2(float(px) + 0.5, float(py) + 0.5);
    float3 ray_dir = pixel_to_camera_ray(
        pixel_center,
        principal_px,
        inv_linear_row0,
        inv_linear_row1,
        radius_to_angle_lut,
        radius_lut_count,
        radius_lut_max_radius,
        radius_lut_tail_slope
    );

    float best_depth = depth.load(int(py), int(px));
    float4 best_color = rgb.load(int(py), int(px));
    bool found_closer_hit = false;

    [loop]
    for (uint near_idx = 0; near_idx < near_triangle_count; ++near_idx)
    {
        uint triangle_index = near_triangle_indices[near_idx];
        TrianglePrimitive tri = triangles[triangle_index];
        float3 v0 = world_to_camera_point(tri.p0.xyz, world_to_camera_rdf);
        float3 v1 = world_to_camera_point(tri.p1.xyz, world_to_camera_rdf);
        float3 v2 = world_to_camera_point(tri.p2.xyz, world_to_camera_rdf);

        float max_depth = max(v0.z, max(v1.z, v2.z));
        if (max_depth <= near_plane_m)
            continue;

        float t;
        float3 bary;
        if (!ray_triangle_intersect(ray_dir, v0, v1, v2, t, bary))
            continue;

        float3 hit = ray_dir * t;
        if (hit.z <= near_plane_m || hit.z >= best_depth)
            continue;

        best_depth = hit.z;
        best_color = apply_distance_fog(tri.color, hit.z, fog_start_m, fog_end_m, fog_power);
        found_closer_hit = true;
    }

    if (found_closer_hit)
    {
        depth.store(int(py), int(px), best_depth);
        depth_bits.store(int(py), int(px), asuint(best_depth));
        rgb.store(int(py), int(px), best_color);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void pack_rgb_to_rgba8(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<uint, 2> rgba8,
    uniform uint image_width,
    uniform uint image_height)
{
    if (dispatchThreadID.x >= image_width || dispatchThreadID.y >= image_height)
        return;

    int px = int(dispatchThreadID.x);
    int py = int(dispatchThreadID.y);
    float4 color = saturate(rgb.load(py, px));
    uint r = uint(color.x * 255.0 + 0.5);
    uint g = uint(color.y * 255.0 + 0.5);
    uint b = uint(color.z * 255.0 + 0.5);
    uint a = uint(color.w * 255.0 + 0.5);
    rgba8.store(py, px, r | (g << 8) | (b << 16) | (a << 24));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_triangles_screen(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<TrianglePrimitive> triangles,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform float fog_start_m,
    uniform float fog_end_m,
    uniform float fog_power)
{
    uint idx = dispatchThreadID.x;
    uint triangle_count;
    uint triangle_stride;
    triangles.GetDimensions(triangle_count, triangle_stride);
    if (idx >= triangle_count)
        return;

    TrianglePrimitive tri = triangles[idx];
    rasterize_projected_triangle(
        tri.p0.xy,
        tri.p0.z,
        tri.p1.xy,
        tri.p1.z,
        tri.p2.xy,
        tri.p2.z,
        tri.color,
        rgb,
        depth,
        depth_bits,
        image_width,
        image_height,
        fog_start_m,
        fog_end_m,
        fog_power
    );
}

[shader("compute")]
[numthreads(8, 8, 1)]
void raytrace_polygon_camera(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float4> polygon_vertices_camera,
    uniform StructuredBuffer<float> radius_to_angle_lut,
    uniform RWTensor<float4, 2> rgb,
    uniform RWTensor<float, 2> depth,
    uniform RWTensor<uint, 2> depth_bits,
    uniform uint image_width,
    uniform uint image_height,
    uniform uint bbox_x0,
    uniform uint bbox_y0,
    uniform uint vertex_offset,
    uniform uint vertex_count,
    uniform float4 plane_camera,
    uniform uint dominant_axis,
    uniform float2 principal_px,
    uniform float2 inv_linear_row0,
    uniform float2 inv_linear_row1,
    uniform float near_plane_m,
    uniform uint lut_count,
    uniform float lut_max_radius,
    uniform float lut_tail_slope,
    uniform float4 color,
    uniform float fog_start_m,
    uniform float fog_end_m,
    uniform float fog_power)
{
    uint px = bbox_x0 + dispatchThreadID.x;
    uint py = bbox_y0 + dispatchThreadID.y;
    if (px >= image_width || py >= image_height)
        return;

    float2 pixel_center = float2(float(px) + 0.5, float(py) + 0.5);
    float3 ray_dir = pixel_to_camera_ray(
        pixel_center,
        principal_px,
        inv_linear_row0,
        inv_linear_row1,
        radius_to_angle_lut,
        lut_count,
        lut_max_radius,
        lut_tail_slope
    );

    float denom = dot(plane_camera.xyz, ray_dir);
    if (abs(denom) <= 1e-6)
        return;

    float t = -plane_camera.w / denom;
    if (t <= 0.0)
        return;

    float3 hit = ray_dir * t;
    if (hit.z <= near_plane_m)
        return;

    if (!point_in_polygon_camera(hit, polygon_vertices_camera, vertex_offset, vertex_count, dominant_axis))
        return;

    float depth_value = hit.z;
    uint depth_value_bits = asuint(depth_value);
    uint old_bits;
    int linear_idx = int(py) * int(depth_bits._strides[0]) + int(px) * int(depth_bits._strides[1]) + int(depth_bits._offset);
    InterlockedMin(depth_bits._data[linear_idx], depth_value_bits, old_bits);
    if (depth_value_bits < old_bits)
    {
        depth.store(int(py), int(px), depth_value);
        rgb.store(int(py), int(px), apply_distance_fog(color, depth_value, fog_start_m, fog_end_m, fog_power));
    }
}
