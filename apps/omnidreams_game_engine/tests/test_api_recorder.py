# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the packet recorder, its reader and the replay source."""

from __future__ import annotations

from pathlib import Path

import pytest
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0.recorder import RX, TX

pytestmark = pytest.mark.ci_cpu


def test_a_log_is_readable_before_close_and_skips_an_unfinished_line(
    tmp_path: Path,
) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(RX, {"type": "frame"}, b"seed", host_sends=0)
    with (tmp_path / "packets.jsonl").open("a", encoding="utf-8") as index:
        index.write('{"seq": 1, "dir": "rx", "hea')
    records = list(v0.PacketLogReader(tmp_path))
    assert [(r.header["type"], r.payload) for r in records] == [("frame", b"seed")]
    recorder.close()


def test_replay_refuses_a_log_with_a_host_initiated_restart(tmp_path: Path) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(TX, {"type": "restart", "reason": "refresh"})
    recorder.close()
    with pytest.raises(ValueError, match="request_refresh"):
        v0.PacketLogSource(tmp_path, v0.FluStateCodec())


def test_recorder_writes_both_directions_and_omits_tx_payloads_by_default(
    tmp_path: Path,
) -> None:
    recorder = v0.PacketRecorder(tmp_path)
    recorder.record(
        RX, {"type": "frame", "epoch": 1, "g": 0, "tick": 0}, b"seed", host_sends=0
    )
    recorder.record(TX, {"type": "gen_frame"}, b"pixels")
    recorder.close()
    records = list(v0.PacketLogReader(tmp_path))
    assert [(r.direction, r.payload, r.host_sends) for r in records] == [
        (RX, b"seed", 0),
        (TX, b"", None),
    ]
    full = v0.PacketRecorder(tmp_path / "full", record_tx_payloads=True)
    full.record(TX, {"type": "gen_frame"}, b"pixels")
    full.close()
    assert [r.payload for r in v0.PacketLogReader(tmp_path / "full")] == [b"pixels"]


def test_bad_logs_and_records_are_refused(tmp_path: Path) -> None:
    for name, text, error in (
        ("empty", "", "is empty"),
        ("other", '{"kind": "meta"}\n', "not a"),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "packets.jsonl").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            v0.PacketLogReader(tmp_path / name)
    recorder = v0.PacketRecorder(tmp_path / "log")
    with pytest.raises(ValueError, match="direction"):
        recorder.record("sideways", {"type": "x"})
    recorder.close()
    recorder.record(RX, {"type": "x"})
    assert recorder.records == 0
