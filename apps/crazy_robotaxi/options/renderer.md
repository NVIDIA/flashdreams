# RENDERER

[Options index](../OPTIONS.md)

The raster is the main semantic camera image. The BEV is the top-down view used by the HUD.

## RASTER

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Width:** | `renderer.raster.width` | Game engine | `--width` | Main raster wbe positiveidth in pixels; must . |
| **Height:** | `renderer.raster.height` | Game engine | `--height` | Main raster height in pixels; must be positive. |
| **Compute Device:** | `renderer.raster.compute_device` | Game engine | — | Device used for raster computation. |
| **Sync Gpu Timing:** | `renderer.raster.sync_gpu_timing` | Game engine | — | Synchronizes GPU work for timing measurements. |
| **Perf Log Interval Frames:** | `renderer.raster.perf_log_interval_frames` | Game engine | — | Interval for raster performance logs. |
| **Near Plane M:** | `renderer.raster.near_plane_m` | Game engine | — | Nearest camera clipping distance; must be less than Far Plane M. |
| **Far Plane M:** | `renderer.raster.far_plane_m` | Game engine | — | Farthest camera clipping distance; must exceed Near Plane M. |
| **Fog Start M:** | `renderer.raster.fog_start_m` | Game engine | — | Distance where fog begins; must be less than Fog End M. |
| **Fog End M:** | `renderer.raster.fog_end_m` | Game engine | — | Distance where fog reaches full strength; must exceed Fog Start M. |
| **Fog Power:** | `renderer.raster.fog_power` | Game engine | — | Exponent controlling the fog transition curve. |
| **Triangle Raytrace Distance M:** | `renderer.raster.triangle_raytrace_distance_m` | Game engine | — | Maximum distance for triangle ray tracing. |
| **Triangle Raytrace Edge Samples:** | `renderer.raster.triangle_raytrace_edge_samples` | Game engine | — | Number of edge samples for triangle ray tracing. |
| **Lane Segment Interval M:** | `renderer.raster.lane_segment_interval_m` | Game engine | — | Segment spacing when rasterizing lanes. |
| **Polyline Segment Interval M:** | `renderer.raster.polyline_segment_interval_m` | Game engine | — | Segment spacing when rasterizing other polylines. |
| **Line Width Px:** | `renderer.raster.line_width_px` | Game engine | — | Rendered road-line width in pixels. |
| **Pole Width Px:** | `renderer.raster.pole_width_px` | Game engine | — | Rendered pole width in pixels. |
| **Dual Line Offset M:** | `renderer.raster.dual_line_offset_m` | Game engine | — | Separation of paired road lines. |
| **Depth Clear M:** | `renderer.raster.depth_clear_m` | Game engine | — | Initial depth-buffer distance. |

## BEV

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `renderer.bev.enabled` | Game engine | — | Displays the top-down view. |
| **Width:** | `renderer.bev.width` | Game engine | — | Top-down image width in pixels; must be positive. |
| **Height:** | `renderer.bev.height` | Game engine | — | Top-down image height in pixels; must be positive. |
| **Height M:** | `renderer.bev.height_m` | Game engine | — | Camera height above the scene; must be positive. |
| **Fov Deg:** | `renderer.bev.fov_deg` | Game engine | — | Top-down camera field of view in degrees. |
| **Tilt Deg:** | `renderer.bev.tilt_deg` | Game engine | — | Top-down camera tilt in degrees. |
