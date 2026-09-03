# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image Edit checkpoint and sampling configuration."""

from dataclasses import dataclass

QWEN_IMAGE_EDIT_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
"""Official Qwen Image Edit 2511 Hugging Face repository."""

QWEN_IMAGE_EDIT_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
"""Immutable checkpoint revision validated by this integration."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenImageEditConfig:
    """Configure one native Qwen Image Edit checkpoint."""

    name: str
    """Stable integration slug."""

    repo_id: str = QWEN_IMAGE_EDIT_REPO_ID
    """Hugging Face repository containing all model components."""

    revision: str = QWEN_IMAGE_EDIT_REVISION
    """Immutable repository revision."""

    num_inference_steps: int = 40
    """Default Euler denoising step count."""

    true_cfg_scale: float = 4.0
    """Classifier-free guidance scale used by the official 2511 example."""

    negative_prompt: str = " "
    """Negative prompt which enables true classifier-free guidance."""


QWEN_IMAGE_EDIT_2511 = QwenImageEditConfig(name="qwen-image-edit-2511")
"""Production Qwen Image Edit 2511 configuration."""
