# Assets

This directory holds the bundled HUD control sprites under
`wheel_and_pedals/` and the map-independent obstacle template catalog.

## `obstacle_vehicle_tracks_v1.npz`

Numeric vehicle trajectories used by the live-edit obstacle ability. The
catalog was deterministically extracted from
`clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz` with:

```bash
python -m crazy_robotaxi.live_edit.obstacle_template_authoring \
  /path/to/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz \
  crazy_robotaxi/assets/obstacle_vehicle_tracks_v1.npz
```

The archive contains only concatenated numeric arrays: relative timestamps,
local center translations, orientations, first-sample dimensions, object-type
codes, sample offsets, and initial height above the source ground. It contains
all 668 Car and Truck tracks from the source obstacle table. Runtime loading
uses `allow_pickle=False`; the source USDZ is authoring input and is not
packaged.

## `wheel_and_pedals/`

AlpaSim-style steering-wheel and pedal PNGs that drive the desktop
HUD chrome (the `SlangPyHudPresenter` steering-wheel + pedal overlay):

- `steering_wheel.png`
- `throttle_pressed.png`, `throttle_unpressed.png`
- `brake_pressed.png`, `brake_unpressed.png`

These are loaded by default (resolved relative to the installed
package), so the realistic controls render out of the box. Pass
`--control-assets-dir` to point the demo at a different sprite set;
the brake PNGs are also accepted under AlpaSim's `break_*.png`
spelling. When a sprite is missing, the HUD falls back to a
CPU-rendered vector wheel / fill-bar pedals.

## Maps

Crazy Robotaxi's `.robotaxi.yaml` maps live under `crazy_robotaxi/maps/`.
Seed images referenced by a map may be map-relative files or packaged assets.
