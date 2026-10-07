# RENDERER

[Options index](../OPTIONS.md)

The raster is the main semantic camera image. The BEV is the top-down view used by the HUD.

## RASTER

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Width:** | Game engine | `--width` | Main raster width in pixels; must be positive. |
| **Height:** | Game engine | `--height` | Main raster height in pixels; must be positive. |
| **Lane Segment Interval M:** | Game engine | — | Segment spacing when rasterizing lanes. |
| **Polyline Segment Interval M:** | Game engine | — | Segment spacing when rasterizing other polylines. |
| **Line Width Px:** | Game engine | — | Rendered road-line width in pixels. |
| **Pole Width Px:** | Game engine | — | Rendered pole width in pixels. |
| **Dual Line Offset M:** | Game engine | — | Separation of paired road lines. |

## BEV

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Game engine | — | Displays the top-down view. |
| **Width:** | Game engine | — | Top-down image width in pixels; must be positive. |
| **Height:** | Game engine | — | Top-down image height in pixels; must be positive. |
| **Height M:** | Game engine | — | Camera height above the scene; must be positive. |
| **Fov Deg:** | Game engine | — | Top-down camera field of view in degrees. |
| **Tilt Deg:** | Game engine | — | Top-down camera tilt in degrees. |
