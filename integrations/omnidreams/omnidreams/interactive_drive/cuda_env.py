# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os

DISABLE_CUDA_INTEROP_ENV = "INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP"

DISABLE_CUDNN_SDP_ENV = "INTERACTIVE_DRIVE_DISABLE_CUDNN_SDP"
"""Route attention away from cuDNN's scaled-dot-product backend.

An escape hatch for hosts where cuDNN itself is unusable -- a sublibrary
version mismatch surfaces as ``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`` on
the first attention call, which no amount of ordering fixes. Torch falls back
to its flash / efficient kernels, so the demo runs somewhat slower rather than
not at all.
"""


def env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_cudnn_sdp_opt_out() -> bool:
    """Disable cuDNN attention when the opt-out is set.

    Returns:
        Whether the backend was disabled, so callers can log it once.
    """
    if not env_truthy(DISABLE_CUDNN_SDP_ENV):
        return False
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
    return True
