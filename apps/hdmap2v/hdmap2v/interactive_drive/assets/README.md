# Assets

This directory holds the unpacked-scene-bundle loader
(`scene_bundle.py`) plus the bundled HUD control sprites under
`wheel_and_pedals/`.

## `wheel_and_pedals/`

AlpaSim-style steering-wheel and pedal PNGs that drive the desktop
HUD chrome (the `SlangPyHudPresenter` steering-wheel + pedal overlay):

- `steering_wheel.png`
- `throttle_pressed.png`, `throttle_unpressed.png`
- `brake_pressed.png`, `brake_unpressed.png`

These are loaded by default (resolved relative to the installed
package), so the realistic controls render out of the box. Pass
`--control-assets-dir` to point the application at a different sprite set;
the brake PNGs are also accepted under AlpaSim's `break_*.png`
spelling. When a sprite is missing, the HUD falls back to a
CPU-rendered vector wheel / fill-bar pedals.

## Scenes

Scene USDZs are staged by the selected model integration, **not** here.
Consult that integration's README for its cache and preparation commands.
