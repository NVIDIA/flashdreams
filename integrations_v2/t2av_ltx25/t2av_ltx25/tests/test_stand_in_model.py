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

"""CPU contracts and synchronized MP4 coverage using a deterministic backend."""

import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from t2av_ltx25 import LTX25Application, create_app
from t2av_ltx25.app import LTX25ModelLoop
from t2av_ltx25.backend import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    BackendLoadConfig,
    GeneratedMedia,
    GenerationRequest,
)

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the synchronized MP4 contract requires ffmpeg and ffprobe on PATH",
)

_PROMPT = "A paper windmill turns beside a chiming bell."
_FPS = 24
_WIDTH = 320
_HEIGHT = 192
_FRAMES = 25


class StandInBackend:
    """Cheap deterministic joint generator with the production backend surface."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.close_count = 0

    @property
    def sample_rate(self) -> int:
        """Return the production checkpoint's PCM rate."""
        return DEFAULT_AUDIO_SAMPLE_RATE

    @property
    def audio_channels(self) -> int:
        """Return the production checkpoint's channel count."""
        return DEFAULT_AUDIO_CHANNELS

    def generate(self, request: GenerationRequest) -> GeneratedMedia:
        """Return moving RGB bars and audible stereo tones for one request."""
        self.requests.append(request)
        video = _stand_in_video(request)
        audio = _stand_in_audio(request)
        return GeneratedMedia(
            video=video,
            audio=audio,
            metrics={
                "stand_in_s": 0.001,
                "audio_samples_count": int(audio.shape[1]),
            },
        )

    def close(self) -> None:
        """Record application-scoped ownership without releasing real resources."""
        self.close_count += 1


class RecordingLoader:
    """Backend loader recording when model construction becomes necessary."""

    def __init__(self, backend: StandInBackend | None = None) -> None:
        self.backend = StandInBackend() if backend is None else backend
        self.configs: list[BackendLoadConfig] = []

    def __call__(self, config: BackendLoadConfig) -> StandInBackend:
        self.configs.append(config)
        return self.backend


def _stand_in_video(request: GenerationRequest) -> torch.Tensor:
    """Return deterministic moving RGB bars in production TCHW layout."""
    x = torch.arange(request.width, dtype=torch.int32).view(1, 1, 1, -1)
    y = torch.arange(request.height, dtype=torch.int32).view(1, 1, -1, 1)
    t = torch.arange(request.num_frames, dtype=torch.int32).view(-1, 1, 1, 1)
    red = (x + 11 * t).expand(-1, 1, request.height, -1)
    green = (y + 7 * t).expand(-1, 1, -1, request.width)
    blue = (x + y + 3 * t).expand(-1, 1, -1, -1)
    return torch.cat((red, green, blue), dim=1).remainder(256).to(torch.uint8)


def _stand_in_audio(request: GenerationRequest) -> torch.Tensor:
    """Return distinct deterministic left and right tones for the clip timeline."""
    sample_count = round(
        request.num_frames * DEFAULT_AUDIO_SAMPLE_RATE / request.frame_rate
    )
    timeline = torch.arange(sample_count, dtype=torch.float32)
    left = torch.sin(2 * math.pi * 440 * timeline / DEFAULT_AUDIO_SAMPLE_RATE)
    right = torch.sin(2 * math.pi * 660 * timeline / DEFAULT_AUDIO_SAMPLE_RATE)
    return 0.2 * torch.stack((left, right))


def _application(loader: RecordingLoader | None = None) -> LTX25Application:
    """Return an application backed by a recording stand-in loader."""
    return LTX25Application(RecordingLoader() if loader is None else loader)


def _initialized_application(
    loader: RecordingLoader | None = None,
    *,
    frames: int = _FRAMES,
) -> LTX25Application:
    """Return a stand-in application initialized with representative arguments."""
    app = _application(loader)
    app.init(
        [
            "--prompt",
            _PROMPT,
            "--num-frames",
            str(frames),
            "--seed",
            "7",
            "--device",
            "cuda:0",
            "--offload",
            "sequential",
            "--local-files-only",
        ]
    )
    return app


def _session_desc(
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
    frames_per_second: int = _FPS,
) -> SessionDesc:
    """Return the lossless synchronized runtime contract used by CPU tests."""
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=frames_per_second,
        video_width=width,
        video_height=height,
        audio_sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        audio_channels=DEFAULT_AUDIO_CHANNELS,
    )


def _step(app: LTX25Application) -> tuple[StepResult, LTX25ModelLoop]:
    """Create a session and run its one joint generation step directly."""
    session = app.create_session(_session_desc())
    session.init()
    results = session.model_loop.step(0, UserInputEvents([]))
    assert isinstance(results, list)
    assert len(results) == 1
    loop = session.model_loop
    assert isinstance(loop, LTX25ModelLoop)
    return results[0], loop


def _probe(path: Path) -> dict[str, Any]:
    """Return FFprobe's stream metadata for an encoded MP4."""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-count_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def _decoded_audio(path: Path) -> np.ndarray:
    """Decode the MP4 audio stream to sample-major stereo float PCM."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(DEFAULT_AUDIO_SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)


def test_factory_returns_a_v2_application() -> None:
    assert isinstance(create_app(), IApplication)


def test_session_desc_is_complete_and_does_not_load_the_backend() -> None:
    loader = RecordingLoader()
    app = _application(loader)

    assert app.session_desc() == SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=24,
        video_width=768,
        video_height=512,
        audio_sample_rate=48_000,
        audio_channels=2,
    )
    assert loader.configs == []


def test_create_session_requires_init() -> None:
    with pytest.raises(RuntimeError, match="init"):
        _application().create_session(_session_desc())


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--prompt", "   "], "--prompt"),
        (["--prompt", _PROMPT, "--seed", "-1"], "--seed"),
        (["--prompt", _PROMPT, "--num-frames", "8"], "8k"),
        (["--prompt", _PROMPT, "--num-frames", "249"], "1 through 241"),
        (["--prompt", _PROMPT, "--device", "   "], "--device"),
    ],
)
def test_init_rejects_invalid_generation_arguments(
    args: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _application().init(args)


@pytest.mark.parametrize(
    ("desc", "message"),
    [
        (replace(_session_desc(), output_layout=VideoTensorLayout.bcthw), "tchw"),
        (replace(_session_desc(), video_width=321), "divisible by 32"),
        (replace(_session_desc(), video_height=193), "divisible by 32"),
        (replace(_session_desc(), audio_sample_rate=24_000), "48000 Hz"),
        (replace(_session_desc(), audio_channels=1), "stereo"),
        (
            replace(_session_desc(), backpressure_mode=BackpressureMode.DROP_OLDEST),
            "blocking backpressure",
        ),
        (
            replace(
                _session_desc(),
                presentation_mode=PresentationMode.ONLY_PRESENT_NEWEST,
            ),
            "only new frames",
        ),
    ],
)
def test_session_contract_is_rejected_before_model_loading(
    desc: SessionDesc, message: str
) -> None:
    loader = RecordingLoader()
    app = _initialized_application(loader)

    with pytest.raises(ValueError, match=message):
        app.create_session(desc)

    assert loader.configs == []


def test_backend_loads_once_and_sessions_have_isolated_state() -> None:
    loader = RecordingLoader()
    app = _initialized_application(loader)
    first = app.create_session(_session_desc())
    second = app.create_session(_session_desc())
    first.init()
    second.init()

    first_result = first.model_loop.step(0, UserInputEvents([]))

    assert len(loader.configs) == 1
    assert first.model_loop.is_finished()
    assert not second.model_loop.is_finished()
    assert isinstance(first_result, list)
    second.model_loop.step(0, UserInputEvents([]))
    assert len(loader.backend.requests) == 2


def test_step_returns_exact_video_audio_and_metrics_contract() -> None:
    loader = RecordingLoader()
    result, loop = _step(_initialized_application(loader))

    assert result.step_index == 0
    assert result.output.shape == (_FRAMES, 3, _HEIGHT, _WIDTH)
    assert result.output.dtype is torch.uint8
    assert result.output.is_contiguous()
    assert result.frame_count == _FRAMES
    assert result.output_layout is VideoTensorLayout.tchw
    assert result.audio is not None
    assert result.audio.samples.shape == (
        2,
        round(_FRAMES * DEFAULT_AUDIO_SAMPLE_RATE / _FPS),
    )
    assert result.audio.samples.dtype is torch.float32
    assert result.audio.sample_rate == DEFAULT_AUDIO_SAMPLE_RATE
    assert result.audio.sample_offset == 0
    assert result.metrics["audio_samples_count"] == result.audio.samples.shape[1]
    assert result.metrics["model_load_s"] >= 0
    assert result.metrics["generation_s"] > 0
    assert result.metrics["generation_fps"] > 0
    assert loop.is_finished()

    request = loader.backend.requests[0]
    assert request == GenerationRequest(
        prompt=_PROMPT,
        seed=7,
        num_frames=_FRAMES,
        width=_WIDTH,
        height=_HEIGHT,
        frame_rate=_FPS,
    )


def test_reset_repeats_the_same_seeded_generation() -> None:
    app = _initialized_application()
    session = app.create_session(_session_desc())
    session.init()
    first = session.model_loop.step(0, UserInputEvents([]))
    assert isinstance(first, list)

    session.model_loop.reset()
    second = session.model_loop.step(0, UserInputEvents([]))

    assert isinstance(second, list)
    assert torch.equal(first[0].output, second[0].output)
    assert first[0].audio is not None
    assert second[0].audio is not None
    assert torch.equal(first[0].audio.samples, second[0].audio.samples)


def test_application_owns_and_closes_the_backend_once() -> None:
    loader = RecordingLoader()
    app = _initialized_application(loader)
    app.create_session(_session_desc())

    app.close()
    app.close()

    assert loader.backend.close_count == 1


@needs_ffmpeg
def test_runtime_writes_decodable_synchronized_audio_video_and_stats(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "ltx25-stand-in.mp4"
    stats_path = tmp_path / "ltx25-stand-in-stats.json"
    loader = RecordingLoader()
    app = _application(loader)

    ApplicationRunner(
        app,
        Mp4ClientWindow(video_path),
        metrics_output_sink=MetricsOutputSink(stats_path),
    ).run(
        _session_desc(),
        ["--prompt", _PROMPT, "--num-frames", str(_FRAMES), "--seed", "7"],
    )

    assert video_path.stat().st_size > 10_000
    probe = _probe(video_path)
    streams = probe["streams"]
    assert isinstance(streams, list)
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert video["codec_name"] == "h264"
    assert int(video["nb_read_frames"]) == _FRAMES
    assert int(video["width"]) == _WIDTH
    assert int(video["height"]) == _HEIGHT
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == DEFAULT_AUDIO_SAMPLE_RATE
    assert int(audio["channels"]) == DEFAULT_AUDIO_CHANNELS

    decoded_audio = _decoded_audio(video_path)
    assert decoded_audio.shape[0] >= _FRAMES * DEFAULT_AUDIO_SAMPLE_RATE // _FPS
    assert np.isfinite(decoded_audio).all()
    assert float(np.abs(decoded_audio).max()) > 0.05

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["steps"] == [
        {"frame_count": _FRAMES, "sample_count": 5, "step_index": 0}
    ]
    samples = {sample["name"]: sample for sample in stats["samples"]}
    assert samples["generation_fps"]["unit"] == "fps"
    assert samples["generation_s"]["unit"] == "s"
    assert samples["audio_samples_count"]["unit"] == "count"
    assert loader.backend.close_count == 1
