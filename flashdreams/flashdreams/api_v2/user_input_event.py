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

"""User input event protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, final

from numpy import uint64


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class UserInputEvent(ABC):
    """Base class for timestamped user input events."""

    _type_name_owners: ClassVar[dict[str, str]] = {}
    """ClassVar tracking all registered UserInputEvent's."""

    timestamp: uint64
    """Timestamp in microseconds since the start of the session."""

    event_id: str | None = None
    """Browser-generated correlation ID; ``None`` for untraced input."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register and validate the concrete event type name.

        Raises:
            TypeError: The event type name is not a non-empty string.
            ValueError: Another event class uses the same type name.
        """
        super(UserInputEvent, cls).__init_subclass__(**kwargs)
        type_name = cls.get_type_name()
        if not isinstance(type_name, str) or not type_name:
            raise TypeError("User input event type names must be non-empty strings.")

        owner = f"{cls.__module__}.{cls.__qualname__}"
        registered_owner = cls._type_name_owners.get(type_name)
        if registered_owner is not None and registered_owner != owner:
            raise ValueError(
                f"User input event type name {type_name!r} is already registered "
                f"by {registered_owner}."
            )
        cls._type_name_owners[type_name] = owner

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        ...

    @final
    def get_timestamp(self) -> uint64:
        """Return the timestamp of the event."""
        return self.timestamp

    def __hash__(self) -> int:
        """Return the hash of the concrete class name.

        The value is not stable across processes.
        """
        return hash(type(self).__name__)
