# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Protocol


class RuntimeControls(Protocol):
    """Live application/UI state read once per loop iteration.

    Distinct from :class:`~omnidreams.interactive_drive.input.backend.InputBackend`,
    which produces continuous driving commands (throttle/brake/steer).
    These are discrete UI affordances: the current view mode and a
    rising-edge reset request.
    """

    @property
    def view_mode(self) -> str: ...

    def consume_reset_request(self) -> bool:
        """Return ``True`` exactly once per :py:meth:`request_reset` call.

        The single-consumer rising-edge contract: the loop calls this
        each iteration; the first call after a reset request returns
        ``True`` and clears the pending flag, subsequent calls return
        ``False`` until a new reset is requested.
        """
        ...
