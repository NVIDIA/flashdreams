# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application sessions."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, final

from flashdreams.api_v2.thread import (
    BlitModelOutputToScreenThread,
    IThread,
    UIThread,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc


class ISession(ABC):
    """One application run with a model thread and a UI thread.

    Register the model thread in :meth:`init`. If no UI thread is registered,
    the runtime uses :class:`BlitModelOutputToScreenThread`.
    """

    _registered_ui_thread: UIThread[Any] | None = None
    _registered_model_thread: IThread[Any] | None = None
    _registrations_frozen = False

    @cached_property
    def _presentation_manager(self) -> PresentationManager:
        """Return this session's model frame buffer."""
        return PresentationManager()

    @abstractmethod
    def init(self) -> None:
        """Initialize state and register the UI and model threads."""
        ...

    @property
    @abstractmethod
    def session_desc(self) -> SessionDesc:
        """Return the description used to configure the runtime."""
        ...

    @final
    def register_ui_thread(
        self,
        thread_type: type[UIThread[Any]],
        *,
        state: Any = None,
        **kwargs: Any,
    ) -> UIThread[Any]:
        """Create and register the UI thread.

        Omit this call to use the default UI.
        """
        if self._registrations_frozen:
            raise RuntimeError("Thread registrations are already in use.")
        if self._registered_ui_thread is not None:
            raise RuntimeError("The session already registered a UI thread.")
        if not issubclass(thread_type, UIThread):
            raise TypeError("The UI thread must derive from UIThread.")
        thread = thread_type(
            state=state,
            frequency=self.session_desc.frames_per_second_for_ui,
            output_layout=self.session_desc.output_layout,
            presentation_manager=self._presentation_manager,
            **kwargs,
        )
        self._registered_ui_thread = thread
        return thread

    @final
    def register_model_thread(
        self,
        thread_type: type[IThread[Any]],
        *,
        state: Any,
        **kwargs: Any,
    ) -> IThread[Any]:
        """Create and register the model thread."""
        if self._registrations_frozen:
            raise RuntimeError("Thread registrations are already in use.")
        if self._registered_model_thread is not None:
            raise RuntimeError("The session already registered a model thread.")
        if not issubclass(thread_type, IThread):
            raise TypeError("The model thread must derive from IThread.")
        if issubclass(thread_type, UIThread):
            raise TypeError("A UIThread cannot be registered as the model thread.")
        thread = thread_type(
            state=state,
            frequency=self.session_desc.frames_per_second_for_step,
            **kwargs,
        )
        self._registered_model_thread = thread
        return thread

    @property
    @final
    def ui_thread(self) -> UIThread[Any]:
        """Return the registered UI thread."""
        if self._registered_ui_thread is None:
            raise RuntimeError("The session has not registered a UI thread.")
        return self._registered_ui_thread

    @property
    @final
    def model_thread(self) -> IThread[Any]:
        """Return the registered model-generation thread."""
        if self._registered_model_thread is None:
            raise RuntimeError("The session has not registered a model thread.")
        return self._registered_model_thread

    def close(self) -> None:
        """Release resources owned by the session."""
        return

    @final
    def _take_threads(self) -> tuple[UIThread[Any], IThread[Any]]:
        """Return both threads and stop further registration."""
        ui_thread = self._registered_ui_thread
        if ui_thread is None:
            ui_thread = self.register_ui_thread(BlitModelOutputToScreenThread)
        model_thread = self._registered_model_thread
        if model_thread is None:
            raise RuntimeError("ISession.init() did not register a model thread.")
        self._registrations_frozen = True
        return ui_thread, model_thread

    @final
    def _shutdown_registered_threads(self) -> list[BaseException]:
        """Close both threads and return any errors."""
        failures: list[BaseException] = []
        for thread in (
            self._registered_model_thread,
            self._registered_ui_thread,
        ):
            if thread is None:
                continue
            try:
                thread._shutdown()
            except BaseException as error:
                failures.append(error)
        return failures


__all__ = ["ISession"]
