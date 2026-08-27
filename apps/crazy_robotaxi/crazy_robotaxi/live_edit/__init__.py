# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OmniDreams live-edit abilities for Crazy Robotaxi (flag-gated, off by default).

Draft scaffold for the integration plan in
``integrations/omnidreams/docs/robotaxi_live_edit_integration.md``. Two
abilities, each behind its own ``--live-edit-*`` flag:

- **Style skins** (:mod:`crazy_robotaxi.live_edit.style_ability`): mid-run
  world restyle via a pre-merged text-edit LoRA + drift corrector attached
  to the flashdreams session, toggled by a prompt swap between chunks.
- **Coin pickups** (:mod:`crazy_robotaxi.live_edit.coin_ability` +
  :mod:`crazy_robotaxi.live_edit.presenter`): sprites laid out along the
  navigation lanes, projected per frame through the scene's FTheta camera,
  composited into the world-model frame before HUD/encode, collected by
  ego proximity.
- **Weather events** (:mod:`crazy_robotaxi.live_edit.weather_ability`, key
  ``v``): clear -> rain -> snow -> clear via plain two-prompt-guided swaps
  (no LoRA; the style LoRA is bypassed for weather-only windows) issued
  through the same :class:`StyleAbility` prompt state machine.
- **Effect items** (:mod:`crazy_robotaxi.live_edit.item_ability`): sparse
  pickup items along the lanes (rain/snow icons, mystery boxes) that reuse
  the coin course/projection/compositing machinery and trigger the
  weather/skin state machines at the next chunk boundary — the same path
  the key requests take (the keys stay live alongside).
- **Obstacle events** (:mod:`crazy_robotaxi.live_edit.obstacle_ability`,
  key ``o``): generated crossing cars with a gameplay-owned lifecycle,
  optional compiled-road placement, optional PhysX materialization, and
  optional box-axis guidance on the model side.

Wiring map (composition-root seams, all live in ``crazy_robotaxi``):

1. CLI: ``runtime_cli.build_parser`` registers the ``--live-edit-*`` flags
   via :func:`crazy_robotaxi.live_edit.config.add_live_edit_args`; the run
   paths build :class:`LiveEditConfig` with
   :func:`crazy_robotaxi.live_edit.config.live_edit_config_from_args` and
   hand it to ``CrazyRobotaxiApp``.
2. Session attach: ``CrazyRobotaxiApp.__init__`` calls
   :func:`crazy_robotaxi.live_edit.style_ability.install_style_ability_on_backend`
   before model warmup starts (corrector-safe session swap + deferred
   LoRA attach).
3. Presenter: ``CrazyRobotaxiApp.__init__`` wraps the chain as
   ``CausalFrameAlignmentPresenter(LiveEditPresenter(inner, ...))`` so the
   compositor sees frame-aligned poses; the per-rollout coin ability is
   bound via ``LiveEditPresenter.set_coin_ability``.
4. Keys: ``k`` cycles the skin, ``c`` toggles coins — native HUD
   (``hud_presenter._on_keyboard_event``) and MJPEG
   (``streaming_presenter._apply_control``) both raise rising-edge
   requests on ``CrazyRobotaxiKeyboardState.live_edit``
   (:class:`crazy_robotaxi.live_edit.input_hooks.LiveEditRequests`),
   drained each tick by ``CrazyRobotaxiRuntime.process_events``.
"""

from crazy_robotaxi.live_edit.config import (
    LiveEditCoinsConfig,
    LiveEditConfig,
    LiveEditItemsConfig,
    LiveEditObstacleConfig,
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    add_live_edit_args,
    live_edit_config_from_args,
)

__all__ = [
    "LiveEditCoinsConfig",
    "LiveEditConfig",
    "LiveEditItemsConfig",
    "LiveEditObstacleConfig",
    "LiveEditStyleConfig",
    "LiveEditWeatherConfig",
    "add_live_edit_args",
    "live_edit_config_from_args",
]
