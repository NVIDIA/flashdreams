# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Revocable controller authority across queued input, model work, and previews."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from handdreams.controller_preview import (
    ControllerPreview,
    ControllerPreviewCalibration,
    trigger_preview,
)
from handdreams.ui import HandDreamsUIState

from apps.handdreams.tests.test_calibration_flow import _frame, _session
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("remote", [False, True])
def test_queued_revoked_input_cannot_reach_model_while_local_input_stays_valid(
    remote, monkeypatch
):
    authorized = [True]
    authority = (lambda: authorized[0]) if remote else None
    loop = _session(auto_start=True, stale_ms=None).model_loop
    try:
        first = replace(_frame(0), authority_valid=authority)
        assert loop.step(0, UserInputEvents([first])) == []
        transform = Mock(wraps=loop.state.pipeline.transform)
        monkeypatch.setattr(loop.state.pipeline, "transform", transform)
        queued = UserInputEvents([replace(_frame(1), authority_valid=authority)])
        authorized[0] = False

        result = loop.step(1, queued)

        assert len(result) == (0 if remote else 1)
        assert transform.call_count == (0 if remote else 1)
        assert len(loop.state.builder.pipeline.generate_calls) == (0 if remote else 1)
        assert loop.state.pipeline.windows == ([] if remote else [[0, 1]])
        assert loop.state.last_input_frame_id == (0 if remote else 1)
        assert len(loop.state.poses) == (0 if remote else 1)
    finally:
        loop.close()


def test_revocation_discards_ready_buffer_without_a_disconnect_event():
    authorized = [True]

    def authority():
        return authorized[0]

    loop = _session(auto_start=True, stale_ms=None).model_loop
    try:
        assert (
            loop.step(
                0, UserInputEvents([replace(_frame(0), authority_valid=authority)])
            )
            == []
        )
        assert (
            len(
                loop.step(
                    1,
                    UserInputEvents(
                        [replace(_frame(i), authority_valid=authority) for i in (1, 2)]
                    ),
                )
            )
            == 1
        )
        assert loop.state.poses.ready(loop.state.actions_per_step)
        assert loop.state.control_sampler is not None
        assert len(loop.state.control_sampler) > 0
        windows = list(loop.state.pipeline.windows)
        authorized[0] = False

        assert loop.step(2, UserInputEvents([])) == []

        assert len(loop.state.poses) == 0
        assert len(loop.state.control_sampler) == 0
        assert loop.state.input_authority_valid is None
        assert loop.state.pipeline.windows == windows
        assert len(loop.state.builder.pipeline.generate_calls) == 1
    finally:
        loop.close()


@pytest.mark.parametrize("stage", ["prepare_window", "encode_actions"])
def test_revocation_during_action_preparation_aborts_before_model_submission(
    stage, monkeypatch
):
    authorized = [True]

    def authority():
        return authorized[0]

    loop = _session(auto_start=True, stale_ms=None).model_loop
    try:
        assert (
            loop.step(
                0, UserInputEvents([replace(_frame(0), authority_valid=authority)])
            )
            == []
        )
        owner = loop.state.pipeline if stage == "prepare_window" else loop.state.builder
        original = getattr(owner, stage)

        def revoke(*args, **kwargs):
            prepared = original(*args, **kwargs)
            authorized[0] = False
            return prepared

        monkeypatch.setattr(owner, stage, revoke)
        assert (
            loop.step(
                1, UserInputEvents([replace(_frame(1), authority_valid=authority)])
            )
            == []
        )

        assert loop.state.pipeline.windows == [[0, 1]]
        assert loop.state.pipeline.aborts == 1
        assert loop.state.pipeline.commits == 0
        assert loop.state.builder.pipeline.generate_calls == []
        assert loop.state.builder.pipeline.finalize_calls == []
        assert loop.state.steps_generated == loop.state.rows_consumed == 0
        assert len(loop.state.poses) == 0
        assert loop.state.input_authority_valid is None
    finally:
        loop.close()


@pytest.mark.parametrize("remote", [False, True])
def test_retained_input_rechecks_authority_for_ui_and_previews(remote):
    authorized = [True]
    event = replace(
        _frame(0), authority_valid=(lambda: authorized[0]) if remote else None
    )
    queued = UserInputEvents([event])
    preview = ControllerPreview(
        ControllerPreviewCalibration.capture(
            event.to_controller_frame(),
            None,
            participant_basis=None,
            position_gain=1.0,
        )
    )
    state = HandDreamsUIState()
    state.observe_events(queued)
    assert state.status.left_tracked and state.status.right_tracked
    assert all(claw is not None for claw in trigger_preview(event))
    assert all(claw is not None for claw in preview.observe(event))
    authorized[0] = False

    state.observe_events(queued)

    assert event.is_authorized() is not remote
    assert event.to_controller_frame()["left"]["tracked"] is not remote
    assert event.to_controller_frame()["right"]["tracked"] is not remote
    assert state.status.left_tracked is not remote
    assert state.status.right_tracked is not remote
    assert state.status.left_open is not remote
    assert state.status.right_open is not remote
    assert (trigger_preview(event) == (None, None)) is remote
    assert (preview.observe(event) == (None, None)) is remote
