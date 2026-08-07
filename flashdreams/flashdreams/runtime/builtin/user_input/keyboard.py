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

"""Raw keyboard-event input contract for the builtin runtime."""

from enum import Enum

from flashdreams.runtime.input_system import RawUserInput


class KeyboardEvent(str, Enum):
    """Keyboard edge types reported by raw input sources."""

    KEY_DOWN = "keydown"
    KEY_UP = "keyup"


class KeyboardKey(str, Enum):
    """Keyboard key identifiers supported by builtin input handlers."""

    W = "w"
    A = "a"
    S = "s"
    D = "d"
    Q = "q"
    E = "e"
    I = "i"
    J = "j"
    K = "k"
    L = "l"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SPACE = "space"


class RawUserKeyboardEvent(RawUserInput):
    """Timestamped raw keyboard edge received from an input source."""

    event: KeyboardEvent
    """Keyboard edge reported by the input source."""

    key: KeyboardKey
    """Supported keyboard key associated with the edge."""
