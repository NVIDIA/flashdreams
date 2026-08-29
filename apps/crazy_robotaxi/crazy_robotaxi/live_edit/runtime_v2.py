# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""API-v2 model-thread wiring for Crazy Robotaxi live-edit abilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.contracts import GameRules, GameUpdate
from omnidreams_game_engine.types import SceneDefinition, TrajectoryChunk, VehicleState
from PIL import Image, ImageDraw
from torch import Tensor

from crazy_robotaxi.live_edit.coin_ability import CoinAbility
from crazy_robotaxi.live_edit.config import ITEM_TYPES, LiveEditConfig
from crazy_robotaxi.live_edit.gpu_compositor import LiveEditFrameCompositor
from crazy_robotaxi.live_edit.item_ability import ItemAbility, ItemEffects
from crazy_robotaxi.live_edit.nitro_ability import NitroAbility
from crazy_robotaxi.live_edit.obstacle_ability import (
    OBSTACLE_ENTITY_PREFIX,
    ObstacleGuidance,
)
from crazy_robotaxi.live_edit.obstacle_events import ObstacleAbility
from crazy_robotaxi.live_edit.style_ability import StyleAbility
from crazy_robotaxi.navigation import NavigationLane
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


def _procedural_coin_sprite() -> Image.Image:
    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (8, 8, 88, 88),
        fill=(250, 200, 40, 255),
        outline=(170, 120, 10, 255),
        width=6,
    )
    draw.ellipse((24, 24, 72, 72), outline=(170, 120, 10, 255), width=4)
    return image


_ITEM_PLACEHOLDERS = {
    "rain": ((70, 130, 240, 255), (20, 60, 160, 255), "R"),
    "snow": ((235, 245, 255, 255), (120, 160, 210, 255), "S"),
    "mystery": ((250, 170, 40, 255), (160, 90, 10, 255), "?"),
    "nitro": ((90, 225, 110, 255), (20, 120, 40, 255), "N"),
}


def _procedural_item_sprite(item_type: str) -> Image.Image:
    fill, rim, label = _ITEM_PLACEHOLDERS[item_type]
    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((9, 9, 87, 87), radius=16, fill=fill, outline=rim, width=6)
    box = draw.textbbox((0, 0), label)
    draw.text(
        ((96 - (box[2] - box[0])) / 2, (96 - (box[3] - box[1])) / 2 - box[1]),
        label,
        fill=rim,
    )
    return image


def _load_sprite(path: Path | None, fallback: Image.Image) -> Image.Image:
    if path is None:
        return fallback
    with Image.open(path) as image:
        return image.convert("RGBA")


class LiveEditGameplay:
    """Own all flag-gated CPU gameplay and presentation state."""

    def __init__(
        self,
        config: LiveEditConfig,
        scene: SceneDefinition,
        lanes: tuple[NavigationLane, ...],
        *,
        vehicle: Any,
    ) -> None:
        self.config = config
        self.style = (
            StyleAbility(config.style, config.weather)
            if config.style.enabled or config.weather.enabled
            else None
        )
        self.coins = (
            CoinAbility.from_lanes(lanes, config.coins)
            if config.coins.enabled
            else None
        )
        self.items = (
            ItemAbility.from_lanes(lanes, config.items)
            if config.items.enabled
            else None
        )
        self.nitro = NitroAbility(config.items) if config.items.enabled else None
        self.effects = (
            ItemEffects(self.style, config.items, nitro_ability=self.nitro)
            if self.items is not None
            else None
        )
        self.obstacles = (
            ObstacleAbility(
                config.obstacle,
                game_map=scene.game_map,
                ground_vertices=scene.ground_mesh_vertices,
                vehicle=vehicle,
            )
            if config.obstacle.enabled
            else None
        )
        self.guidance = (
            ObstacleGuidance(config.obstacle.guide_scale)
            if config.obstacle.enabled and config.obstacle.guide_scale > 0.0
            else None
        )
        sprites = (
            {
                item_type: _load_sprite(
                    config.items.sprite_path(item_type),
                    _procedural_item_sprite(item_type),
                )
                for item_type in ITEM_TYPES
            }
            if config.items.enabled
            else {}
        )
        self._compositor = LiveEditFrameCompositor(
            _load_sprite(config.coins.sprite_path, _procedural_coin_sprite()), sprites
        )
        self._camera = FThetaCameraModel(
            scene.selected_camera,
            output_width=scene.initial_rgb.shape[1],
            output_height=scene.initial_rgb.shape[0],
        )
        self._frame_index = 0

    def attach_model(self, pipeline: Any) -> None:
        """Install optional model-side obstacle guidance."""
        if self.guidance is not None:
            self.guidance.install_v2(pipeline)

    def adopt_model_state(
        self,
        previous: LiveEditGameplay,
        pipeline: Any,
        cache: Any,
        base_prompt: str,
    ) -> None:
        """Reuse installed model hooks while resetting per-rollout gameplay."""
        if self.style is not None and previous.style is not None:
            self.style = previous.style
            self.style.reset_v2(cache)
            if self.items is not None:
                self.effects = ItemEffects(
                    self.style,
                    self.config.items,
                    nitro_ability=self.nitro,
                )
        if self.guidance is not None and previous.guidance is not None:
            self.guidance = previous.guidance
            self.guidance.reset_v2(pipeline)

    def prepare_model_step(
        self, pipeline: Any, engine: Any, step: Any, autoregressive_index: int
    ) -> None:
        """Prepare the obstacle-free shadow conditioning branch."""
        if self.guidance is None:
            return
        actors = step.trajectory.dynamic_actors
        filtered = tuple(
            actor
            for actor in actors
            if not actor.entity_id.startswith(OBSTACLE_ENTITY_PREFIX)
        )
        alternate = None
        if len(filtered) != len(actors):
            clean_trajectory = replace(step.trajectory, dynamic_actors=filtered)
            alternate = engine.condition_renderer.render(clean_trajectory).hdmap_bvtchw
        self.guidance.prepare_v2(
            pipeline,
            autoregressive_index,
            step.condition.hdmap_bvtchw,
            alternate,
        )

    @property
    def actor_controllers(self) -> tuple[ObstacleAbility, ...]:
        """Return physical obstacle control when that mode is enabled."""
        if self.obstacles is None or not self.config.obstacle.physics:
            return ()
        return (self.obstacles,)

    def process_events(self, events: UserInputEvents) -> None:
        """Consume rising-edge ability keys on the V2 model thread."""
        for event in events.get_events():
            if not isinstance(event, KeyboardUserInputEvent):
                continue
            if event.state is not KeyboardInputState.PRESSED:
                continue
            key = str(event.key).strip().lower()
            if key == "k" and self.style is not None:
                self.style.request_cycle()
            elif key == "v" and self.style is not None:
                self.style.request_weather_cycle()
            elif key == "c" and self.coins is not None:
                self.coins.toggle()
            elif key == "o" and self.obstacles is not None:
                self.obstacles.request_spawn()

    def advance(self, trajectory: TrajectoryChunk) -> tuple[Any, ...]:
        """Advance pickups and obstacles after one physics trajectory."""
        if self.coins is not None:
            self.coins.advance_frames(trajectory.vehicle_states)
        if self.items is not None and self.effects is not None:
            for item_type in self.items.advance_frames(trajectory.vehicle_states):
                self.items.flash(self.effects.apply(item_type))
        if self.obstacles is None:
            return ()
        return self.obstacles.advance_frames(trajectory)

    def postprocess_video(self, video: Tensor, step: Any) -> Tensor:
        """Composite frame-aligned collectibles and state chips on device."""
        if (
            self.coins is None
            and self.items is None
            and self.style is None
            and self.obstacles is None
        ):
            return video
        result = video.clone()
        _, _, frame_count = result.shape[:3]
        for index in range(frame_count):
            pose = step.trajectory.rig_poses_world[index]
            sprites = []
            if self.coins is not None:
                sprites.extend(
                    self.coins.visible_sprites(
                        pose,
                        self._camera,
                        image_width=int(result.shape[-1]),
                        image_height=int(result.shape[-2]),
                    )
                )
            if self.items is not None:
                sprites.extend(
                    self.items.visible_sprites(
                        pose,
                        self._camera,
                        image_width=int(result.shape[-1]),
                        image_height=int(result.shape[-2]),
                    )
                )
            labels = []
            if self.style is not None:
                labels.append(f"SKIN {self.style.active_skin_name.upper()}")
                labels.append(f"WEATHER {self.style.active_weather_name.upper()}")
            if self.coins is not None:
                labels.append(
                    f"COINS {self.coins.collected_count}  +{self.coins.score}"
                )
            if self.nitro is not None and self.nitro.active:
                labels.append(f"NITRO {self.nitro.seconds_remaining:.1f}s")
            if self.items is not None and self.items.flash_label is not None:
                labels.append(self.items.flash_label)
            if self.obstacles is not None:
                labels.append(
                    f"OBSTACLES {len(self.obstacles.events)}  HITS {self.obstacles.hit_count}"
                )
            frame = ((result[0, 0, index] + 1.0) * 127.5).round().clamp(0, 255)
            frame = frame.to(torch.uint8).permute(1, 2, 0).contiguous()
            frame = self._compositor.composite(
                frame,
                sprites=sprites,
                frame_index=self._frame_index,
                labels=labels,
                sharpen_sigma=self.config.sharpen_sigma,
                sharpen_amount=(
                    self.config.sharpen_amount
                    if self.style is not None and self.style.active_skin_name != "base"
                    else 0.0
                ),
            )
            if self.config.obstacle.annotate:
                frame = self._annotate_obstacles(
                    frame,
                    pose,
                    int(step.trajectory.timestamps_us[index]),
                )
            result[0, 0, index] = (
                frame.permute(2, 0, 1).to(dtype=result.dtype) / 127.5 - 1.0
            )
            self._frame_index += 1
        return result

    def _annotate_obstacles(
        self, frame: Tensor, pose: Any, timestamp_us: int
    ) -> Tensor:
        """Draw the optional obstacle 3D-box evidence overlay."""
        if self.obstacles is None or not self.obstacles.events:
            return frame
        device = frame.device
        canvas = Image.fromarray(frame.cpu().numpy(), mode="RGB")
        draw = ImageDraw.Draw(canvas)
        signs = np.asarray(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            dtype=np.float32,
        )
        edges = (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for event in self.obstacles.events:
            center = event.center_at(timestamp_us)
            orientation = event.orientation_at(timestamp_us)
            if center is None or orientation is None:
                continue
            x, y, z, w = (float(value) for value in orientation)
            rotation = np.asarray(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )
            half = np.asarray(event.dimensions_lwh, dtype=np.float32) / 2.0
            corners = center[None, :] + (signs * half[None, :]) @ rotation.T
            uv, _depth, forward = self._camera.project_world(corners, pose)
            if not forward.all():
                continue
            for first, second in edges:
                draw.line(
                    [tuple(uv[first].tolist()), tuple(uv[second].tolist())],
                    fill=(255, 60, 60),
                    width=2,
                )
        array = np.asarray(canvas, dtype=np.uint8).copy()
        return torch.from_numpy(array).to(device=device)


class LiveEditGameRules:
    """Add live-edit dynamic actors while preserving the selected game rules."""

    def __init__(self, inner: GameRules, gameplay: LiveEditGameplay) -> None:
        self.inner = inner
        self.gameplay = gameplay

    @property
    def is_running(self) -> bool:
        return self.inner.is_running

    def snapshot(self, vehicle_state: VehicleState) -> object:
        return self.inner.snapshot(vehicle_state)

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> GameUpdate:
        update = self.inner.advance_frames(trajectory, frame_interval_s)
        return GameUpdate(
            frames=update.frames,
            dynamic_actors=(*update.dynamic_actors, *self.gameplay.advance(trajectory)),
        )

    def submit_text(self, value: str, vehicle_state: VehicleState) -> object:
        return self.inner.submit_text(value, vehicle_state)


__all__ = ["LiveEditGameRules", "LiveEditGameplay"]
