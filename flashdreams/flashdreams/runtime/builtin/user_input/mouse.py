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

"""Raw mouse-button event contract for the builtin runtime."""

from enum import Enum

from flashdreams.runtime.input_system import RawUserInput


class MouseEvent(str, Enum):
    """Mouse-button edge types reported by raw input sources."""

    BUTTON_DOWN = "mousedown"
    BUTTON_UP = "mouseup"


class MouseButton(str, Enum):
    """Mouse buttons supported by builtin input handlers."""

    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class RawUserMouseEvent(RawUserInput):
    """Timestamped raw mouse-button edge received from an input source."""

    event: MouseEvent
    """Mouse-button edge reported by the input source."""

    button: MouseButton
    """Mouse button associated with the edge."""
