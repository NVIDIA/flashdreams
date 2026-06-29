# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams.interactive_drive.scene_loader import (
    load_scene_bundle as load_scene_bundle,
)


def canonicalize_camera_name(name: str) -> str:
    return name.strip().lower().replace(":", "_").replace("-", "_")


__all__ = [
    "canonicalize_camera_name",
    "load_scene_bundle",
]
