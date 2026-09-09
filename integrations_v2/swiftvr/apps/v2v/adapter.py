# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
