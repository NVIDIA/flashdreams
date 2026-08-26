# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

from cam2v import Cam2VApplication, Cam2VApplicationDefaults
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3

from flashdreams.api_v2.application import IApplication

from .conditioning import resolve_lingbot_conditioning

_INSTALL_HINT = (
    "Install the Lingbot Cam2V application: pip install flashdreams-cam2v-lingbot."
)


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World configured through its existing interactive runner config."""

    def __init__(self) -> None:
        super().__init__(
            defaults=Cam2VApplicationDefaults.from_runner_config(
                RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
                input_resolver=resolve_lingbot_conditioning,
                install_hint=_INSTALL_HINT,
            )
        )


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


__all__ = ["LingbotCam2VApplication", "create_app"]
