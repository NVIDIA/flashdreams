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

"""Causal-Forcing text-to-video application factory."""

from t2v import T2VApplication, T2VApplicationDefaults
from causal_forcing.config import PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE
from flashdreams.demo import IFlashDreamsApplication
def create_app() -> IFlashDreamsApplication:
    """Create the Causal-Forcing text-to-video application."""
    return T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
            total_blocks=60,
            pixel_height=480,
            pixel_width=832,
            fps=16,
            output_layout="tchw",
        )
    )
__all__ = ["create_app"]
