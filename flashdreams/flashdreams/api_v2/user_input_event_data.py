# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input event data protocol."""

from abc import ABC, abstractmethod


class UserInputEventData(ABC):
    """Base class for data stored in a user input event.

    Implementations provide :meth:`get_type_name` and may add fields for their
    event data. The runtime owns the set of concrete types, in
    :mod:`flashdreams.runtime_v2.user_input_event`, which covers the input
    modalities supported today.
    """

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """Return the event type name."""
        ...

    @classmethod
    def __hash__(cls) -> int:
        """Return the hash of the concrete class name.

        The value is not stable across processes.
        """
        return hash(str(cls.__name__))
