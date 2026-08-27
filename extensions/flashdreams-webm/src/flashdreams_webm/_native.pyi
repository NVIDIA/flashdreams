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

"""Type declarations for the native WebM extension."""

from os import PathLike
from typing import Any

class WebmWriter:
    """Native incremental VP8/VP9 and Opus WebM writer."""

    def __init__(
        self,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        width: int,
        height: int,
        frames_per_second: int,
        codec: str = "vp9",
        audio_sample_rate: int = 0,
        audio_channels: int = 0,
    ) -> None: ...
    @property
    def codec(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def write_video(self, frames: Any) -> None: ...
    def close(
        self,
        audio_path: str | bytes | PathLike[str] | PathLike[bytes] | None = None,
    ) -> None: ...
    def abort(self) -> None: ...

def versions() -> dict[str, str]: ...
