# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application sessions."""

import queue
import threading
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, final

from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc


class ISession(ABC):
    """One application run with a model loop and a UI loop.

    Register the model loop in :meth:`init`. If no UI loop is registered,
    the runtime uses :class:`BlitModelOutputToScreenLoop`.
    """

    _registered_ui_loop: IUILoop[Any] | None = None
    _registered_model_loop: IModelLoop[Any] | None = None
    _registrations_frozen = False

    @cached_property
    def _shutdown_event(self) -> threading.Event:
        """Return the event shared by all registered loops."""
        return threading.Event()

    @cached_property
    def _failure_queue(self) -> queue.Queue[BaseException]:
        """Return the failure queue shared by all registered loops."""
        return queue.Queue()

    @cached_property
    def _presentation_manager(self) -> PresentationManager:
        """Return this session's model frame buffer."""
        return PresentationManager()

    @abstractmethod
    def init(self) -> None:
        """Initialize state and register this session's loops.

        A model loop is required. Registering a UI loop is optional; without one
        the runtime uses :class:`BlitModelOutputToScreenLoop`.

        Raises:
            RuntimeError: No model loop was registered by the time the runtime
                asks for the loops.
        """
        ...

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to configure the runtime."""
        ...

    @final
    def register_ui_loop(
        self,
        loop_type: type[IUILoop[Any]],
        *,
        state: Any = None,
        **kwargs: Any,
    ) -> IUILoop[Any]:
        """Create and register the UI loop, and return it.

        Omit this call to use the default UI. The loop is paced at the session's
        ``frames_per_second_for_ui``.

        Args:
            loop_type: UI loop class to instantiate.
            state: State the loop owns. Unlike a model loop, a UI loop is
                allowed to hold none.
            **kwargs: Passed to ``loop_type``.

        Raises:
            RuntimeError: The runtime already took the loops, or a UI loop was
                registered already.
            TypeError: ``loop_type`` does not derive from :class:`IUILoop`.
        """
        if self._registrations_frozen:
            raise RuntimeError("Loop registrations are already in use.")
        if self._registered_ui_loop is not None:
            raise RuntimeError("The session already registered a UI loop.")
        if not issubclass(loop_type, IUILoop):
            raise TypeError("The UI loop must derive from IUILoop.")
        loop = loop_type(**kwargs)
        loop.register_session_loop_objects(
            state=state,
            frequency=self.session_desc.frames_per_second_for_ui,
            shutdown_event=self._shutdown_event,
            failure_queue=self._failure_queue,
        )
        loop.register_session_ui_loop_objects(
            output_layout=self.session_desc.output_layout,
            presentation_manager=self._presentation_manager,
        )
        self._registered_ui_loop = loop
        return loop

    @final
    def register_model_loop(
        self,
        loop_type: type[IModelLoop[Any]],
        *,
        state: Any,
        **kwargs: Any,
    ) -> IModelLoop[Any]:
        """Create and register the model loop, and return it.

        The loop is paced at the session's ``frames_per_second_for_step``.

        Args:
            loop_type: Model loop class to instantiate.
            state: State the loop owns. Required, since a model loop with no
                state has nothing to generate from.
            **kwargs: Passed to ``loop_type``.

        Raises:
            RuntimeError: The runtime already took the loops, or a model loop
                was registered already.
            TypeError: ``loop_type`` does not derive from :class:`IModelLoop`.
        """
        if self._registrations_frozen:
            raise RuntimeError("Loop registrations are already in use.")
        if self._registered_model_loop is not None:
            raise RuntimeError("The session already registered a model loop.")
        if not issubclass(loop_type, IModelLoop):
            raise TypeError("The model loop must derive from IModelLoop.")
        loop = loop_type(**kwargs)
        loop.register_session_loop_objects(
            state=state,
            frequency=self.session_desc.frames_per_second_for_step,
            shutdown_event=self._shutdown_event,
            failure_queue=self._failure_queue,
        )
        self._registered_model_loop = loop
        return loop

    @property
    @final
    def ui_loop(self) -> IUILoop[Any]:
        """Return the registered UI loop."""
        if self._registered_ui_loop is None:
            raise RuntimeError("The session has not registered a UI loop.")
        return self._registered_ui_loop

    @property
    @final
    def model_loop(self) -> IModelLoop[Any]:
        """Return the registered model-generation loop."""
        if self._registered_model_loop is None:
            raise RuntimeError("The session has not registered a model loop.")
        return self._registered_model_loop

    def close(self) -> None:
        """Release resources owned by the session."""
        return

    @final
    def _take_loops(self) -> tuple[IUILoop[Any], IModelLoop[Any]]:
        """Return both loops and stop further registration."""
        ui_loop = self._registered_ui_loop
        if ui_loop is None:
            ui_loop = self.register_ui_loop(BlitModelOutputToScreenLoop)
        model_loop = self._registered_model_loop
        if model_loop is None:
            raise RuntimeError("ISession.init() did not register a model loop.")
        ui_loop._set_model_loop(model_loop)
        self._registrations_frozen = True
        return ui_loop, model_loop

    @final
    def _shutdown_registered_loops(self) -> list[BaseException]:
        """Request shutdown, close every registered loop, and return errors."""
        self._shutdown_event.set()
        failures: list[BaseException] = []
        for loop in (
            self._registered_model_loop,
            self._registered_ui_loop,
        ):
            if loop is None:
                continue
            try:
                loop._shutdown()
            except BaseException as error:
                failures.append(error)
        return failures


__all__ = ["ISession"]
