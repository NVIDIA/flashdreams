# Assets

This directory holds the bundled HUD control sprites under
`wheel_and_pedals/`.

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
