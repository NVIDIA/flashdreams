# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR binding for the reusable v2v application."""

from v2v import V2VApplication, V2VApplicationDefaults

from flashdreams.api_v2.application import IApplication
from swiftvr.impl.postprocess import POSTPROCESS_PRESET_SWIFTVR_2X


def create_app() -> IApplication:
    """Create the SwiftVR 2x V2V application."""
    processor = POSTPROCESS_PRESET_SWIFTVR_2X
    return V2VApplication(
        defaults=V2VApplicationDefaults(
            processor=processor,
            first_chunk_size=processor.chunk_size,
            steady_chunk_size=processor.chunk_size,
            model_name="swiftvr-2x",
        )
    )


__all__ = ["create_app"]
