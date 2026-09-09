# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU integration checks for native H3 sampling and the shared v2 lifecycle."""

from dataclasses import replace
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flashdreams.runtime_v2.user_input_events import UserInputEvents
from minimax_h3.apps.t2v.adapter import MiniMaxH3Application
from minimax_h3.impl.layout import MiniMaxH3DenoiseState
from minimax_h3.impl.pipeline import MiniMaxH3Pipeline, MiniMaxH3PipelineConfig
from minimax_h3.impl.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
def test_v2_workflows_reset_and_finish_without_loading_models(
    monkeypatch, tmp_path, workflow
):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)
    app = MiniMaxH3Application(MiniMaxH3PipelineConfig(workflow=workflow))
    args = ["--device", "cpu", "--prompt", "motion", "--seed", "7", "--steps", "3"]
    if workflow == "fl2va":
        args += ["--last-image-path", str(image)]
    if workflow == "ref2va":
        args += ["--reference", f"image:{image}"]
    app.init(args)
    pipeline = app._pipeline
    stages = []
    monkeypatch.setattr(
        pipeline, "_condition", lambda request: stages.append("condition") or {}
    )
    monkeypatch.setattr(
        pipeline,
        "_denoise",
        lambda request, condition: stages.append("denoise") or torch.zeros(1),
    )
    monkeypatch.setattr(
        pipeline.decoder,
        "forward",
        lambda latent: stages.append("decode") or torch.zeros(124, 3, 32, 32),
    )
    desc = replace(app.session_desc(), video_width=32, video_height=32)
    session = app.create_session(desc)
    session.init()
    loop = session.model_loop
    first_cache = loop.state.cache
    assert pipeline.config.seed == 7
    assert pipeline.config.scheduler.num_inference_steps == 3
    assert first_cache.num_frames == 124
    result = loop.step(0, UserInputEvents([]))[0]
    assert result.frame_count == 124
    assert result.read_output().shape == (124, 3, 32, 32)
    assert loop.is_finished()
    assert stages == ["condition", "denoise", "decode"]
    assert set(result.metrics) == {
        "conditioning_seconds",
        "denoise_seconds",
        "decode_seconds",
        "total_seconds",
    }
    loop.reset()
    assert not loop.is_finished()
    assert loop.state.cache is not first_cache
    assert loop.state.cache.references == first_cache.references
    assert loop.state.cache.last_image_path == first_cache.last_image_path
    loop.step(0, UserInputEvents([]))
    assert stages == ["condition", "denoise", "decode"] * 2
    loop.close()
    app.close()


def test_joint_sampler_keeps_conditioning_and_calls_network_once(monkeypatch):
    from minimax_h3.impl import layout

    state = MiniMaxH3DenoiseState(
        latents=torch.tensor([[9.0, 9.0], [0.0, 0.0], [0.0, 0.0]]),
        audio_latents=torch.tensor([[8.0, 8.0], [0.0, 0.0]]),
        prompt_embeds=torch.zeros(1, 1, 4),
        position_ids=torch.zeros(6, 3),
        token_tags=torch.tensor([0, 0, 0, 2, 2, 1]),
        video_indices=torch.tensor([0, 1, 2]),
        audio_indices=torch.tensor([3, 4]),
        text_indices=torch.tensor([5]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_latent_frames=2,
        latent_height=1,
        latent_width=1,
    )
    calls = []

    class JointModel(torch.nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            torch.testing.assert_close(kwargs["hidden_states"][0, 0], state.latents[0])
            torch.testing.assert_close(
                kwargs["audio_hidden_states"][0, 0], state.audio_latents[0]
            )
            rows = kwargs["timestep"][kwargs["timestep_indices"]]
            assert rows[0] >= 0.999
            assert rows[3] == 1
            return torch.ones_like(kwargs["hidden_states"]), torch.ones_like(
                kwargs["audio_hidden_states"]
            )

    monkeypatch.setattr(layout, "prepare_denoise_state", lambda *args: state)
    monkeypatch.setattr(MiniMaxH3TransformerConfig, "setup", lambda self: JointModel())
    config = MiniMaxH3PipelineConfig(
        transformer=MiniMaxH3TransformerConfig(in_channels=2, patch_size=(1, 1, 1))
    )
    pipeline = MiniMaxH3Pipeline(config)
    request = pipeline.initialize_cache(text=["motion"], height=2, width=2)
    result = pipeline._denoise(request, {})
    assert len(calls) == 29
    assert result.shape == (1, 2, 2, 1, 1)
    torch.testing.assert_close(result, torch.ones_like(result))
    torch.testing.assert_close(state.latents[0], torch.tensor([9.0, 9.0]))


def test_decode_normalization_and_layout(monkeypatch):
    from minimax_h3.impl.pipeline import MiniMaxH3Decoder

    class Decoder:
        config = SimpleNamespace(latents_mean=(2.0,), latents_std=(3.0,))

        def decode(self, value):
            torch.testing.assert_close(value, torch.full_like(value, 5))
            return torch.zeros(1, 3, 2, 4, 4)

    decoder = MiniMaxH3Decoder(SimpleNamespace(video_decoder=lambda device: Decoder()))
    result = decoder(torch.ones(1, 1, 2, 1, 1))
    assert result.shape == (2, 3, 4, 4)
    torch.testing.assert_close(
        result[0, :, 0, 0], torch.tensor([0.485, 0.456, 0.406]) * 2 - 1
    )


def test_decoder_failure_releases_module(monkeypatch):
    import weakref
    from minimax_h3.impl import pipeline as module

    references = []

    class Decoder:
        config = SimpleNamespace(latents_mean=(0.0,), latents_std=(1.0,))

        def decode(self, value):
            raise RuntimeError("decode failed")

    def load(device):
        decoder = Decoder()
        references.append(weakref.ref(decoder))
        return decoder

    decoder = module.MiniMaxH3Decoder(SimpleNamespace(video_decoder=load))
    with pytest.raises(RuntimeError, match="decode failed"):
        decoder(torch.zeros(1, 1, 2, 1, 1))
    import gc

    gc.collect()
    assert references[0]() is None


@pytest.mark.parametrize(
    "args",
    [
        ["--total-blocks", "2"],
        ["--steps", "1"],
        ["--duration", "nan"],
        ["--lora-scale", "nan"],
    ],
)
def test_bad_arguments_fail_before_weight_loading(args):
    app = MiniMaxH3Application()
    with pytest.raises(ValueError):
        app.init(["--device", "cpu", "--prompt", "motion", *args])


def test_native_applications_import_with_diffusers_blocked():
    code = """
import importlib.abc, sys
class BlockDiffusers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "diffusers" or fullname.startswith("diffusers."):
            raise AssertionError("Diffusers runtime import: " + fullname)
sys.meta_path.insert(0, BlockDiffusers())
from minimax_h3.apps.t2v.adapter import create_app, create_app_fl2va, create_app_ref2va
from minimax_h3.impl.video_vae import VideoVAEConfig
from minimax_h3.impl.audio_encoder import AudioEncoderConfig
for factory in (create_app, create_app_fl2va, create_app_ref2va):
    app = factory()
    assert app.session_desc().frames_per_second_for_step == 24
app = create_app()
app.init(["--device", "cpu", "--prompt", "native"])
session = app.create_session(app.session_desc())
session.init()
session.model_loop.reset()
app.close()
assert not any(k == "diffusers" or k.startswith("diffusers.") for k in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=os.environ.copy(), timeout=90
    )
