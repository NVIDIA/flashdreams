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
`--control-assets-dir` to point the demo at a different sprite set;
the brake PNGs are also accepted under AlpaSim's `break_*.png`
spelling. When a sprite is missing, the HUD falls back to a
CPU-rendered vector wheel / fill-bar pedals.

## Scenes

Scene USDZs themselves are staged into the shared `omnidreams` scene
cache under `$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/`, **not** here.
See `omnidreams.scenes` and `omnidreams-prepare` for how staging
works; both the desktop demo and the WebRTC server consume from the
same cache root.
