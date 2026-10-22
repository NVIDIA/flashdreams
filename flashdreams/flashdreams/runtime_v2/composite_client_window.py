# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window composed with secondary input sources and output sinks."""

from collections.abc import Callable, Sequence

from flashdreams.api_v2.client_window import ClientWindowLink, IClientWindow
from flashdreams.api_v2.input_source import InputSource
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class CompositeClientWindowError(RuntimeError):
    """Failure from one or more components during a lifecycle operation."""

    def __init__(self, operation: str, errors: Sequence[BaseException]) -> None:
        self.operation = operation
        self.errors = tuple(errors)
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(
            f"CompositeClientWindow.{operation} failed for {len(errors)} "
            f"component(s): {details}"
        )


class CompositeClientWindow(IClientWindow):
    """Compose one primary window with secondary input and output components."""

    def __init__(
        self,
        primary_window: IClientWindow,
        secondary_input_sources: Sequence[InputSource] = (),
        secondary_output_sinks: Sequence[OutputSink] = (),
        *,
        on_open: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
        on_ui_step_completed: Callable[[], None] | None = None,
    ) -> None:
        self._primary_window = primary_window
        self._secondary_input_sources = tuple(secondary_input_sources)
        self._secondary_output_sinks = tuple(secondary_output_sinks)
        self._on_open = on_open
        self._on_close = on_close
        self._on_ui_step_completed = on_ui_step_completed

    @property
    def primary_window(self) -> IClientWindow:
        """Return the primary client window."""
        return self._primary_window

    @property
    def secondary_input_sources(self) -> tuple[InputSource, ...]:
        """Return secondary input sources in polling order."""
        return self._secondary_input_sources

    @property
    def secondary_output_sinks(self) -> tuple[OutputSink, ...]:
        """Return secondary output sinks in write order."""
        return self._secondary_output_sinks

    def get_user_input_events(self) -> UserInputEvents:
        """Return primary input followed by each secondary source's input."""
        events = self._primary_window.get_user_input_events().get_events()
        for source in self._secondary_input_sources:
            events.extend(source.get_user_input_events().get_events())
        return UserInputEvents(events)

    def ui_step_completed(self) -> None:
        """Notify interested secondary components after a UI-loop step."""
        if self._on_ui_step_completed is not None:
            self._on_ui_step_completed()

    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Delegate clickable regions only to the primary window."""
        self._primary_window.request_links(links)

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Delegate cursor visibility only to the primary window."""
        self._primary_window.request_hide_cursor(hide_cursor)

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Delegate cursor locking only to the primary window."""
        self._primary_window.request_lock_cursor_to_window(lock_cursor_to_window)

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Delegate window resizing only to the primary window."""
        self._primary_window.request_new_window_size(new_window_size)

    def open(self, session_desc: SessionDesc) -> None:
        """Open outputs in order and roll back opened components on failure."""
        opened: list[OutputSink] = []
        input_lifecycle_started = False
        try:
            self._primary_window.open(session_desc)
            opened.append(self._primary_window)
            for sink in self._secondary_output_sinks:
                sink.open(session_desc)
                opened.append(sink)
            if self._on_open is not None:
                input_lifecycle_started = True
                self._on_open()
        except Exception as error:
            errors = [error]
            if input_lifecycle_started and self._on_close is not None:
                try:
                    self._on_close()
                except Exception as cleanup_error:
                    errors.append(cleanup_error)
            for sink in reversed(opened):
                try:
                    sink.close()
                except Exception as cleanup_error:
                    errors.append(cleanup_error)
            if len(errors) == 1:
                raise
            raise CompositeClientWindowError("open", errors) from error

    def write(self, result: StepResult) -> None:
        """Write in order and stop at the first component failure.

        Earlier components may already have received ``result`` when a later
        component fails.
        """
        self._primary_window.write(result)
        for sink in self._secondary_output_sinks:
            sink.write(result)

    def close(self) -> None:
        """Close every component in reverse open order."""
        errors: list[BaseException] = []
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception as error:
                errors.append(error)
        for sink in (*reversed(self._secondary_output_sinks), self._primary_window):
            try:
                sink.close()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise CompositeClientWindowError("close", errors)


__all__ = ["CompositeClientWindow", "CompositeClientWindowError"]
