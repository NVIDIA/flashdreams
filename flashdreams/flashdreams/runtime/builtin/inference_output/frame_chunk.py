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

"""Tensor frame-chunk output contract for the builtin runtime."""

from typing import Annotated

from flashdreams.runtime.inference_session import InferenceOutput
from pydantic import Field
from torch import Tensor


class FrameChunkOutput(InferenceOutput):
    """Output containing a generated frame chunk."""

    value: Tensor
    """Generated frame chunk with integration-specific tensor layout."""

    start_timestamp: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    """Timestamp of the first frame in seconds on the presentation timeline."""

    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    """Frame rate used to present the chunk."""

    @property
    def frame_present_time(self) -> float:
        """Return the presentation duration of one frame in seconds."""
        return 1.0 / self.fps
