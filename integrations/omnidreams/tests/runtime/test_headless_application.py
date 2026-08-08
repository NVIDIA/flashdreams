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


"""CPU tests for the headless Omnidreams application."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm_euler import (
    FlowMatchEulerDiscreteSchedulerConfig,
)
from flashdreams.infra.encoder import Encoder
from flashdreams.infra.encoder.text.cosmos_reason1 import (
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoder, TeahvVAEDecoderConfig
from flashdreams.recipes.taehv.impl import TAEHVCache
from flashdreams.runtime.builtin.application.video_output_application import (
    VideoOutputApplicationConfig,
)
from flashdreams.runtime.builtin.inference_output.handler.video_output_handler import (
    VideoOutputHandler,
)
from omnidreams.encoder.pixel_shuffle import PixelShuffleVAEEncoderConfig
from omnidreams.pipeline import OmnidreamsPipeline, OmnidreamsPipelineConfig
from omnidreams.runtime.application import headless as headless_module
from omnidreams.runtime.application.headless import (
    OmnidreamsHeadless,
    OmnidreamsHeadlessConfig,
    OmnidreamsInferenceRuntime,
    OmnidreamsInferenceRuntimeConfig,
)
from omnidreams.runtime.global_condition import GlobalConditionHandler
from omnidreams.runtime.inference_session import (
    InferenceGlobalCondition,
    InferenceSession,
    InferenceUserCondition,
)
from omnidreams.runtime.user_input import hdmap_input_handler
from omnidreams.runtime.user_input.hdmap_input_handler import HDMapInputHandler
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.transformer.impl.network import CosmosDiTNetworkConfig
from omnidreams.vae_native import OmnidreamsWanVAEEncoderConfig
from torch import Tensor

pytestmark = pytest.mark.ci_cpu


@dataclass(kw_only=True)
class _TextEncoderConfig(CosmosReason1TextEncoderConfig):
    """Configure the checkpoint-free text encoder used by the CPU pipeline."""

    _target: type["_TextEncoder"] = field(default_factory=lambda: _TextEncoder)


class _TextEncoder(Encoder):
    """Produce deterministic text embeddings without loading a checkpoint."""

    def __init__(self, config: CosmosReason1TextEncoderConfig) -> None:
        """Initialize the stateless encoder contract."""
        super().__init__(config)

    def forward(self, prompts: list[str]) -> Tensor:
        """Return one fixed-size embedding sequence per prompt."""
        return torch.ones(len(prompts), 2, 4)


@dataclass(kw_only=True)
class _ImageEncoderConfig(OmnidreamsWanVAEEncoderConfig):
    """Configure the checkpoint-free image encoder used by the CPU pipeline."""

    _target: type["_ImageEncoder"] = field(default_factory=lambda: _ImageEncoder)


class _ImageEncoder(Encoder):
    """Produce deterministic first-frame latents without loading a checkpoint."""

    def __init__(self, config: OmnidreamsWanVAEEncoderConfig) -> None:
        """Initialize the stateless encoder contract."""
        super().__init__(config)

    def forward(self, image: Tensor) -> Tensor:
        """Pool pixels and expand them to the Omnidreams latent-channel count."""
        pooled = image.mean(dim=(-3, -2, -1), keepdim=True)
        return pooled.expand(*image.shape[:3], 16, 1, 1)


@dataclass(kw_only=True)
class _CPUDecoderConfig(TeahvVAEDecoderConfig):
    """Configure the checkpoint-free decoder used by the CPU pipeline."""

    _target: type["_CPUDecoder"] = field(default_factory=lambda: _CPUDecoder)


class _CPUDecoder(TeahvVAEDecoder):
    """Preserve the concrete TAEHV contract without loading decoder weights."""

    def __init__(self, config: TeahvVAEDecoderConfig) -> None:
        """Initialize only the streaming decoder interface."""
        StreamingVideoDecoder.__init__(self, config)

    def initialize_autoregressive_cache(self) -> TAEHVCache:
        """Return an empty TAEHV-compatible cache."""
        return TAEHVCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: TAEHVCache | None = None,
    ) -> Tensor:
        """Expose three latent channels as a cheap decoded video."""
        del autoregressive_index, cache
        return input[..., :3, :, :]


def _pipeline_config() -> OmnidreamsPipelineConfig:
    """Build an actual Omnidreams pipeline from tiny CPU components."""
    return OmnidreamsPipelineConfig(
        name="test-omnidreams-headless",
        text_encoder=_TextEncoderConfig(),
        image_encoder=_ImageEncoderConfig(),
        encoder=PixelShuffleVAEEncoderConfig(),
        decoder=_CPUDecoderConfig(),
        diffusion_model=DiffusionModelConfig(
            transformer=CosmosTransformerConfig(
                network=CosmosDiTNetworkConfig(
                    in_channels=16,
                    out_channels=16,
                    patch_spatial=1,
                    patch_temporal=1,
                    model_channels=12,
                    num_blocks=0,
                    num_heads=1,
                    mlp_ratio=1.0,
                    concat_padding_mask=False,
                    use_adaln_lora=False,
                    use_crossattn_projection=False,
                    crossattn_emb_channels=4,
                    additional_concat_ch=192,
                ),
                dtype=torch.float32,
                checkpoint_path=None,
                batch_shape=(1,),
                num_views=1,
                len_t=2,
                h_extrapolation_ratio=1.0,
                w_extrapolation_ratio=1.0,
                window_size_t=2,
                sink_size_t=0,
                compile_network=False,
                use_cuda_graph=False,
                skip_finalize_kv_cache=True,
            ),
            scheduler=FlowMatchEulerDiscreteSchedulerConfig(
                num_inference_steps=1,
                fixed_timesteps=(1000.0, 0.0),
            ),
            seed=0,
        ),
    )


def _runtime_config() -> OmnidreamsInferenceRuntimeConfig:
    """Build the production runtime around the checkpoint-free CPU pipeline."""
    return OmnidreamsInferenceRuntimeConfig(
        pipeline=_pipeline_config(),
        session_type=InferenceSession,
        device="cpu",
    )


def _patch_hdmap_video(monkeypatch: pytest.MonkeyPatch, *, num_frames: int) -> None:
    """Patch HDMap decoding with an in-memory RGB video."""
    video = torch.zeros(num_frames, 2, 3, 3, dtype=torch.uint8)

    def normalize_video(
        value: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Convert the patched THWC video into normalized TCHW layout."""
        return value.permute(0, 3, 1, 2).to(device=device, dtype=dtype)

    monkeypatch.setattr(
        hdmap_input_handler,
        "read_video_rgb",
        lambda _path, **_kwargs: video,
    )
    monkeypatch.setattr(
        hdmap_input_handler,
        "rgb_video_to_normalized_tensor",
        normalize_video,
    )
    monkeypatch.setattr(
        hdmap_input_handler,
        "resize_rgb_video",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        headless_module,
        "_load_first_frame",
        lambda _path, **_kwargs: torch.full((1, 1, 1, 3, 8, 8), 0.5),
    )


def test_omnidreams_headless_config_defaults(tmp_path: Path) -> None:
    """Verify direct construction exposes stable rollout defaults."""
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        hdmap_path=tmp_path / "hdmap.mp4",
        first_frame_path=tmp_path / "first.png",
    )

    assert config.artifact_path == headless_module.DEFAULT_ARTIFACT_PATH
    assert config.example_data is False
    assert config.example_data_uuid == headless_module.DEFAULT_EXAMPLE_DATA_UUID
    assert config.text_prompt == headless_module.DEFAULT_TEXT_PROMPT
    assert config.negative_text_prompt == headless_module.NEGATIVE_PROMPT
    assert config.num_frames is None
    assert config.num_chunks == headless_module.DEFAULT_NUM_CHUNKS
    assert config.pixel_height == headless_module.DEFAULT_VIDEO_HEIGHT
    assert config.pixel_width == headless_module.DEFAULT_VIDEO_WIDTH


def test_omnidreams_headless_resolves_example_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify direct construction resolves bundled inputs before setup."""
    _patch_hdmap_video(monkeypatch, num_frames=13)
    hdmap_path = tmp_path / "example_hdmap.mp4"
    first_frame_path = tmp_path / "first_frame.png"
    requested_uuids: list[str] = []

    def download_example(uuid: str) -> tuple[Path, Path]:
        """Record the requested UUID and return local test assets."""
        requested_uuids.append(uuid)
        return hdmap_path, first_frame_path

    monkeypatch.setattr(
        headless_module,
        "download_single_view_example_data",
        download_example,
    )
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=tmp_path / "generated.mp4",
        example_data=True,
        example_data_uuid="test-scene",
        num_chunks=2,
    )

    application = OmnidreamsHeadless(config)

    assert requested_uuids == ["test-scene"]
    assert config.hdmap_path == hdmap_path
    assert config.first_frame_path == first_frame_path
    input_handler = application._user_input_handler
    assert isinstance(input_handler, HDMapInputHandler)
    assert input_handler.hdmap_video_path == hdmap_path


def test_omnidreams_headless_requires_paths_without_example_data() -> None:
    """Verify missing explicit inputs fail before runtime construction."""
    config = OmnidreamsHeadlessConfig(inference_runtime=_runtime_config())

    with pytest.raises(ValueError, match="example_data is enabled"):
        OmnidreamsHeadless(config)


@pytest.mark.parametrize(
    ("num_frames", "num_chunks"),
    [(13, None), (None, 2)],
)
def test_omnidreams_headless_initializes_hdmap_and_video_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    num_frames: int | None,
    num_chunks: int | None,
) -> None:
    """Verify construction configures all production application handlers."""
    _patch_hdmap_video(monkeypatch, num_frames=13)
    artifact_path = tmp_path / "generated.mp4"
    hdmap_path = tmp_path / "hdmap.mp4"
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=artifact_path,
        hdmap_path=hdmap_path,
        first_frame_path=tmp_path / "first.png",
        num_frames=num_frames,
        num_chunks=num_chunks,
    )

    # The specialized integration config extends the reusable video config.
    assert isinstance(config, VideoOutputApplicationConfig)
    assert config._target is OmnidreamsHeadless

    application = OmnidreamsHeadless(config)

    # Runtime setup follows the production construction path and owns an actual
    # Omnidreams pipeline instead of a manually assembled pipeline shell.
    assert type(application._inference_runtime) is OmnidreamsInferenceRuntime
    assert type(application._inference_runtime._pipeline) is OmnidreamsPipeline

    # The production input handler is set up during application initialization
    # with the pipeline's chunk sizing, placement, and rollout limit.
    input_handler = application._user_input_handler
    assert type(input_handler) is HDMapInputHandler
    assert input_handler.hdmap_video_path == hdmap_path
    assert input_handler._get_num_frames(0) == 5
    assert input_handler._get_num_frames(1) == 8
    assert input_handler._num_chunks == 2
    assert input_handler._hdmap.device == torch.device("cpu")
    assert input_handler._hdmap.dtype == torch.float32

    # Construction embeds the config-owned prompt and first frame before
    # releasing the one-shot encoders.
    assert type(application._global_condition_handler) is GlobalConditionHandler
    embedded_condition = application._inference_global_condition
    assert isinstance(embedded_condition, InferenceGlobalCondition)
    assert embedded_condition.text_embeddings.shape == (1, 1, 2, 4)
    assert embedded_condition.negative_text_embeddings is not None
    assert embedded_condition.negative_text_embeddings.shape == (1, 1, 2, 4)
    assert embedded_condition.image_embeddings.shape == (1, 1, 1, 16, 1, 1)
    assert application._inference_runtime._pipeline.text_encoder is None
    assert application._inference_runtime._pipeline.image_encoder is None

    # The reusable parent binds the artifact destination to its video handler.
    output_handler = application._inference_output_handler
    assert isinstance(output_handler, VideoOutputHandler)
    assert output_handler.artifact_path == artifact_path


@pytest.mark.parametrize(
    ("num_frames", "num_chunks"),
    [(13, None), (None, 2)],
)
def test_omnidreams_headless_limits_hdmap_input_during_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    num_frames: int | None,
    num_chunks: int | None,
) -> None:
    """Verify both limit modes produce exactly two complete HDMap chunks."""
    _patch_hdmap_video(monkeypatch, num_frames=21)
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=tmp_path / "generated.mp4",
        hdmap_path=tmp_path / "hdmap.mp4",
        first_frame_path=tmp_path / "first.png",
        num_frames=num_frames,
        num_chunks=num_chunks,
    )
    application = OmnidreamsHeadless(config)
    input_handler = application._user_input_handler

    first = input_handler()
    second = input_handler()

    assert isinstance(first, InferenceUserCondition)
    assert isinstance(second, InferenceUserCondition)
    assert first.hdmap.shape == (1, 1, 5, 3, 2, 3)
    assert second.hdmap.shape == (1, 1, 8, 3, 2, 3)
    with pytest.raises(StopIteration):
        input_handler()


@pytest.mark.parametrize(
    ("num_frames", "num_chunks", "message"),
    [
        (None, None, "exactly one"),
        (13, 2, "exactly one"),
        (0, None, "num_frames must be a positive integer"),
        (None, -1, "num_chunks must be a positive integer"),
    ],
)
def test_omnidreams_headless_rejects_invalid_rollout_limits(
    tmp_path: Path,
    num_frames: int | None,
    num_chunks: int | None,
    message: str,
) -> None:
    """Verify application initialization requires one positive rollout limit."""
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=tmp_path / "generated.mp4",
        hdmap_path=tmp_path / "hdmap.mp4",
        first_frame_path=tmp_path / "first.png",
        num_frames=num_frames,
        num_chunks=num_chunks,
    )

    with pytest.raises(ValueError, match=message):
        OmnidreamsHeadless(config)


def test_omnidreams_headless_rejects_non_boundary_frame_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify an exact frame limit cannot split an autoregressive chunk."""
    _patch_hdmap_video(monkeypatch, num_frames=21)
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=tmp_path / "generated.mp4",
        hdmap_path=tmp_path / "hdmap.mp4",
        first_frame_path=tmp_path / "first.png",
        num_frames=12,
        num_chunks=None,
    )

    with pytest.raises(ValueError, match="next boundary is 13"):
        OmnidreamsHeadless(config)


def test_omnidreams_headless_rejects_short_hdmap_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify construction fails when the HDMap cannot supply every chunk."""
    _patch_hdmap_video(monkeypatch, num_frames=20)
    config = OmnidreamsHeadlessConfig(
        inference_runtime=_runtime_config(),
        artifact_path=tmp_path / "generated.mp4",
        hdmap_path=tmp_path / "hdmap.mp4",
        first_frame_path=tmp_path / "first.png",
        num_chunks=3,
    )

    with pytest.raises(ValueError, match="requires 21 HDMap frames.*contains 20"):
        OmnidreamsHeadless(config)


def test_headless_cli_builds_config_and_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify argparse values drive the concrete application configuration."""
    config_name = headless_module._headless_config_names()[0]
    monkeypatch.setitem(
        headless_module.OMNIDREAMS_CONFIGS,
        config_name,
        _pipeline_config(),
    )
    events: list[str] = []
    recorded: dict[str, object] = {}

    class _CLIApplication:
        """Record CLI application construction and execution."""

        def __init__(self, config: OmnidreamsHeadlessConfig) -> None:
            """Store the application config produced by argument parsing."""
            recorded["config"] = config

        def run(self) -> None:
            """Record application execution."""
            events.append("run")

    monkeypatch.setattr(headless_module, "OmnidreamsHeadless", _CLIApplication)
    artifact_path = tmp_path / "generated.mp4"
    hdmap_path = tmp_path / "hdmap.mp4"
    first_frame_path = tmp_path / "first.png"
    exit_status = headless_module.main(
        [
            "--config",
            config_name,
            "--artifact-path",
            str(artifact_path),
            "--hdmap-path",
            str(hdmap_path),
            "--first-frame-path",
            str(first_frame_path),
            "--text-prompt",
            "drive through a city",
            "--negative-text-prompt",
            "blurry",
            "--device",
            "cpu",
            "--num-chunks",
            "2",
        ]
    )

    assert exit_status == 0
    assert events == ["run"]

    application_config = recorded["config"]
    assert isinstance(application_config, OmnidreamsHeadlessConfig)
    assert application_config.artifact_path == artifact_path
    assert application_config.hdmap_path == hdmap_path
    assert application_config.first_frame_path == first_frame_path
    assert application_config.example_data is False
    assert application_config.text_prompt == "drive through a city"
    assert application_config.negative_text_prompt == "blurry"
    assert application_config.num_frames is None
    assert application_config.num_chunks == 2

    runtime_config = application_config.inference_runtime
    assert isinstance(runtime_config, OmnidreamsInferenceRuntimeConfig)
    assert runtime_config.device == "cpu"
    assert runtime_config.pipeline.name == "test-omnidreams-headless"
    assert (
        runtime_config.pipeline is not headless_module.OMNIDREAMS_CONFIGS[config_name]
    )


def test_headless_cli_accepts_runtime_input_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify example-data selection and the chunk alias reach CLI dispatch."""
    recorded: dict[str, object] = {}
    config_name = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"

    monkeypatch.setattr(
        headless_module,
        "_run_from_args",
        lambda args: recorded.update(vars(args)),
    )

    exit_status = headless_module.main(
        [
            "--config",
            config_name,
            "--device",
            "cuda:0",
            "--example-data",
            "True",
            "--example-data-uuid",
            "239560dc-33d1-11ef-9720-00044bcbccac",
            "--total-blocks",
            "60",
        ]
    )

    assert exit_status == 0
    assert recorded["config"] == config_name
    assert recorded["device"] == "cuda:0"
    assert recorded["num_chunks"] == 60
    assert recorded["artifact_path"] is None
    assert recorded["hdmap_path"] is None
    assert recorded["first_frame_path"] is None
    assert recorded["example_data"] is True
    assert recorded["example_data_uuid"] == "239560dc-33d1-11ef-9720-00044bcbccac"


def test_headless_cli_uses_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify omitted optional arguments retain runtime defaults."""
    recorded: dict[str, object] = {}
    config_name = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"

    monkeypatch.setattr(
        headless_module,
        "_run_from_args",
        lambda args: recorded.update(vars(args)),
    )

    assert (
        headless_module.main(
            [
                "--config",
                config_name,
                "--hdmap-path",
                str(tmp_path / "hdmap.mp4"),
                "--first-frame-path",
                str(tmp_path / "first.png"),
            ]
        )
        == 0
    )
    assert recorded["text_prompt"] == headless_module.DEFAULT_TEXT_PROMPT
    assert recorded["negative_text_prompt"] == headless_module.NEGATIVE_PROMPT
    assert recorded["pixel_height"] == headless_module.DEFAULT_VIDEO_HEIGHT
    assert recorded["pixel_width"] == headless_module.DEFAULT_VIDEO_WIDTH
    assert recorded["num_frames"] is None
    assert recorded["num_chunks"] is None
    assert recorded["example_data"] is False
    assert recorded["example_data_uuid"] == headless_module.DEFAULT_EXAMPLE_DATA_UUID
