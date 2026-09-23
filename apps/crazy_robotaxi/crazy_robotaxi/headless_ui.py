# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HUD-free UI loop that presents raw model frames for headless runs."""

from __future__ import annotations

from torch import Tensor

from crazy_robotaxi.ui import TaxiHudState
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)


class CrazyRobotaxiHeadlessUILoop(BlitModelOutputToScreenLoop[TaxiHudState]):
    """Blit the generated video channel without drawing the ImGui HUD.

    The HUD state still receives every ``invoke_async`` message from the model
    thread (menu selection, loading status, published frames), so the startup
    flow driven by ``--game-mode`` and ``--map`` runs unchanged. Only the video
    channel is presented; the HD-map and BEV debug channels are dropped.
    Nothing here needs a Vulkan device, which makes ``--mode mp4`` runs
    possible on hosts without a SlangPy-compatible graphics adapter (for
    example WSL2).
    """

    def frames_to_blit(self) -> tuple[Tensor, ...]:
        """Return the generated video channel alone."""
        return self.presented_model_frames()[:1]

    def reset(self) -> None:
        """Reset the HUD state the model thread publishes into."""
        self.state.reset()
        super().reset()


__all__ = ["CrazyRobotaxiHeadlessUILoop"]
