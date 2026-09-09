# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for H3 presentation, layouts and staged conditioning."""

from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import gc
import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from minimax_h3.impl.conditioning import (
    _build_presentation,
    _sample_video_condition_frames,
    condition_request,
    encode_visual_condition,
)
from minimax_h3.impl.layout import patchify_video_latents, prepare_denoise_state
from minimax_h3.impl.references import MiniMaxH3Reference

pytestmark = pytest.mark.ci_cpu


class _ImageProcessor:
    merge_size = 2

    def __call__(self, *, images, return_tensors):
        return {
            "pixel_values": torch.zeros(len(images), 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _VideoProcessor:
    temporal_patch_size = 2

    def __call__(self, *, videos, do_sample_frames, return_tensors):
        assert not do_sample_frames
        return {
            "pixel_values_videos": torch.zeros(len(videos), 4),
            "video_grid_thw": torch.tensor(
                [[(len(video) + 1) // 2, 2, 2] for video in videos]
            ),
        }


def _stage_factories(*, fail=None):
    events, live, calls = [], set(), {}

    class Encoder:
        def __init__(self, kind):
            assert not live, f"Overlapping encoder stages: {live} and {kind}"
            live.add(kind)
            events.append(("load", kind))
            self.kind = kind
            weakref.finalize(self, released, kind)
            channels = 32 if kind == "audio" else 24
            self.config = SimpleNamespace(
                latents_mean=[0.0] * channels, latents_std=[1.0] * channels
            )
            self.tokenizer = _Tokenizer()
            self.processor = SimpleNamespace(
                image_processor=_ImageProcessor(), video_processor=_VideoProcessor()
            )

        def sample(self, pixels, *, generator):
            if fail == self.kind:
                raise RuntimeError("injected encoder failure")
            calls.setdefault("video_shapes", []).append(tuple(pixels.shape))
            frames = pixels.shape[2]
            latent_frames = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
            return torch.zeros(
                (1, 24, latent_frames, pixels.shape[3] // 16, pixels.shape[4] // 16)
            )

        def encode(self, waveform):
            if fail == self.kind:
                raise RuntimeError("injected encoder failure")
            calls.setdefault("audio_shapes", []).append(tuple(waveform.shape))
            return torch.zeros((2, 32, 4))

        def __call__(self, input):
            if fail == self.kind:
                raise RuntimeError("injected encoder failure")
            calls["presentation"] = input
            return torch.zeros((1, len(input["token_ids"]), 4))

    def released(kind):
        live.remove(kind)
        events.append(("release", kind))

    return (
        {
            f"{kind}_encoder_factory": lambda kind=kind: Encoder(kind)
            for kind in ("video", "audio", "qwen")
        },
        events,
        live,
        calls,
    )


@pytest.mark.parametrize("anchors", [("first",), ("last",), ("first", "last")])
def test_full_fl2va_request_and_layout(anchors):
    factories, events, live, calls = _stage_factories()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "frame.png"
        Image.new("RGB", (48, 32), (128, 0, 64)).save(path)
        conditioned = condition_request(
            prompt="test",
            workflow="fl2va",
            width=32,
            height=32,
            num_frames=124,
            image_path=path if "first" in anchors else None,
            last_image_path=path if "last" in anchors else None,
            **factories,
        )
    state = prepare_denoise_state(conditioned, 7, "fl2va")
    assert conditioned["keyframe_anchors"] == anchors
    assert calls["video_shapes"] == [(1, 3, 1, 32, 32)] * len(anchors)
    assert state.num_condition_video_rows == len(anchors)
    assert state.num_condition_audio_rows == 0
    assert state.latents.shape == (37 + len(anchors), 96)
    assert state.audio_latents.shape == (414, 32)
    assert not live
    assert events == [
        ("load", "video"),
        ("release", "video"),
        ("load", "qwen"),
        ("release", "qwen"),
    ]


def test_full_mixed_ref2va_request_and_layout(monkeypatch):
    import minimax_h3.impl.conditioning as conditioning

    # Inject the media-decoding boundary, retaining real normalization and packing.
    references = [
        MiniMaxH3Reference(kind="image", image=Image.new("RGB", (32, 32))),
        MiniMaxH3Reference(kind="audio", audio=torch.zeros(1, 3200), sample_rate=32000),
        MiniMaxH3Reference(
            kind="video",
            frames=np.zeros((22, 32, 32, 3), dtype=np.uint8),
            fps=24,
            audio=torch.zeros(2, 3200),
            sample_rate=32000,
        ),
    ]
    monkeypatch.setattr(conditioning, "load_references", lambda specs: references)
    factories, events, live, calls = _stage_factories()
    conditioned = condition_request(
        prompt="mix",
        workflow="ref2va",
        width=32,
        height=32,
        num_frames=124,
        **factories,
    )
    state = prepare_denoise_state(conditioned, 19, "ref2va")
    assert calls["video_shapes"] == [(1, 3, 1, 2048, 2048), (1, 3, 22, 768, 768)]
    assert calls["audio_shapes"] == [(2, 1, 3200), (2, 1, 3200)]
    assert [ref.kind for ref in conditioned["normalized_references"]] == [
        "image",
        "audio",
        "video",
    ]
    assert state.num_condition_video_rows == 64 * 64 + 7 * 24 * 24
    assert state.num_condition_audio_rows == 16
    ids = calls["presentation"]["token_ids"]
    text = bytes(token for token in ids if token < 256).decode()
    assert text == "<Picture 1>: <Audio 1>: <Audio 2>: <Video 1>: <0.2 seconds>mix"
    assert not live
    assert events == [
        ("load", "video"),
        ("release", "video"),
        ("load", "audio"),
        ("release", "audio"),
        ("load", "qwen"),
        ("release", "qwen"),
    ]
    all_indices = torch.cat(
        [state.text_indices, state.video_indices, state.audio_indices]
    )
    assert torch.equal(all_indices.sort().values, torch.arange(len(all_indices)))


@pytest.mark.parametrize("fail", ["video", "qwen"])
def test_conditioning_failure_releases_stage_and_stops(fail):
    factories, events, live, _ = _stage_factories(fail=fail)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "frame.png"
        Image.new("RGB", (32, 32)).save(path)
        try:
            condition_request(
                prompt="test",
                workflow="fl2va",
                width=32,
                height=32,
                num_frames=124,
                image_path=path,
                **factories,
            )
        except RuntimeError as error:
            assert str(error) == "injected encoder failure"
        else:
            pytest.fail("Expected the injected encoder failure")
    # Tracebacks temporarily retain the failing call's local parameter.
    gc.collect()
    assert not live
    assert events[-1] == ("release", fail)
    assert ("load", "audio") not in events
    if fail == "video":
        assert ("load", "qwen") not in events


@pytest.mark.parametrize("anchors", [(), ("first",), ("last",), ("first", "last")])
def test_optional_packed_layout_matches_reference(anchors):
    oracle = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.before_denoise"
    )
    from minimax_h3.impl.layout import build_packed_sequence

    args = (torch.tensor([1, 0, 1]), 37, 6, 10, 207, (1, 2, 2), 2, 2, 0, anchors)
    expected = oracle.MiniMaxH3PrepareLayoutStep.build_packed_sequence(*args)
    for actual, expected in zip(build_packed_sequence(*args), expected):
        assert (
            torch.equal(actual, expected)
            if isinstance(actual, torch.Tensor)
            else actual == expected
        )


def test_optional_mixed_reference_layout_matches_reference():
    oracle = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.before_denoise"
    )
    from minimax_h3.impl.layout import build_ref2va_packed_sequence

    references = [
        MiniMaxH3Reference(kind="image"),
        MiniMaxH3Reference(kind="audio", audio=torch.ones(2, 8)),
        MiniMaxH3Reference(kind="video", audio=torch.ones(2, 8)),
    ]
    args = (
        torch.tensor([1, 0, 1]),
        references,
        [torch.zeros(1, 24, 1, 4, 8), torch.zeros(1, 24, 12, 6, 10)],
        [torch.zeros(16, 32), torch.zeros(24, 32)],
        37,
        6,
        10,
        207,
        (1, 2, 2),
        2,
        2,
        0,
    )
    expected = oracle.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        *args
    )
    for actual, expected in zip(build_ref2va_packed_sequence(*args), expected):
        assert (
            torch.equal(actual, expected)
            if isinstance(actual, torch.Tensor)
            else actual == expected
        )


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text.encode())}

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": 1000,
            "<|vision_end|>": 1001,
            "<|image_pad|>": 1002,
            "<|video_pad|>": 1003,
        }[token]


def test_presentation_preserves_order_modalities_and_verbatim_prompt():
    refs = [
        MiniMaxH3Reference(kind="video", audio=torch.ones(2, 3)),
        MiniMaxH3Reference(kind="image"),
        MiniMaxH3Reference(kind="audio", audio=torch.ones(2, 3)),
    ]
    ids, tags = _build_presentation(_Tokenizer(), " hello ", refs, [2], [3], [[0.25]])
    expected = (
        list(b"<Audio 1>: <Video 1>: <0.2 seconds>")
        + [1000, 1003, 1003, 1003, 1001]
        + list(b"<Picture 1>: ")
        + [1000, 1002, 1002, 1001]
        + list(b"<Audio 2>:  hello ")
    )
    assert ids == expected
    assert tags == [0 if token >= 1000 else 1 for token in ids]


def test_video_sampling_timestamp_rounding_and_minimum():
    frames = np.zeros((37, 2, 2, 3), dtype=np.uint8)
    sampled, timestamps = _sample_video_condition_frames(frames, 24, 2, 2)
    assert len(sampled) == 4
    assert timestamps == [0.25, 1.25]
    with pytest.raises(ValueError):
        _sample_video_condition_frames(frames[:12], 24, 2, 2)


def test_noise_draw_order_condition_rounding_and_layout():
    condition = torch.ones((1, 24, 1, 2, 2))
    conditioned = dict(
        num_frames=124,
        height=32,
        width=32,
        text_token_tags=torch.tensor([1, 0]),
        prompt_embeds=torch.zeros(1, 2, 4),
        keyframe_anchors=("last",),
        condition_latents=[condition],
        audio_condition_latents=[],
    )
    state = prepare_denoise_state(conditioned, 7, "fl2va")
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(condition.shape, generator=generator)
    t = torch.tensor(0.999)
    assert torch.equal(
        state.latents[:1],
        patchify_video_latents(t * condition + (1 - t) * noise, (1, 2, 2)),
    )
    video = torch.randn((1, 24, 37, 2, 2), generator=generator)
    assert torch.equal(state.latents[1:], patchify_video_latents(video, (1, 2, 2)))
    assert torch.equal(state.audio_latents, torch.randn((414, 32), generator=generator))
    assert state.position_ids.dtype == torch.float64
    assert state.num_condition_video_rows == 1
    all_indices = torch.cat(
        [state.text_indices, state.audio_indices, state.video_indices]
    )
    assert sorted(all_indices.tolist()) == list(range(len(all_indices)))


def test_visual_condition_uses_independent_seed_and_fp16_rounding():
    class Encoder:
        config = SimpleNamespace(latents_mean=[0.2], latents_std=[0.3])

        def sample(self, pixels, *, generator):
            assert generator.initial_seed() == 42
            expected = (
                torch.zeros(3) - torch.tensor([0.485, 0.456, 0.406])
            ) / torch.tensor([0.229, 0.224, 0.225])
            assert torch.equal(pixels[0, :, 0, 0, 0], expected)
            return torch.full((1, 1, 1, 1, 1), 0.1234567)

    actual = encode_visual_condition(
        Encoder(), torch.zeros((1, 3, 1, 2, 2), dtype=torch.uint8)
    )
    expected = (torch.tensor(0.1234567).half().float() - 0.2) / 0.3
    assert torch.equal(actual.flatten()[0], expected)


def test_t2va_only_loads_qwen():
    class Qwen:
        tokenizer = _Tokenizer()
        processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))

        def __call__(self, input):
            assert input == {"token_ids": list(b" prompt "), "vision_inputs": {}}
            return torch.zeros(1, 8, 4)

    result = condition_request(
        prompt=" prompt ",
        workflow="t2va",
        width=32,
        height=32,
        num_frames=124,
        qwen_encoder_factory=Qwen,
    )
    assert result["condition_latents"] == []
    assert result["keyframe_anchors"] == ()
    assert result["text_token_tags"].tolist() == [1] * 8


def test_qwen_headless_checkpoint_and_raw_intermediate_state():
    from transformers import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
    )
    from flashdreams.infra.encoder.text.qwen3_vl import (
        Qwen3VLEncoder,
        Qwen3VLEncoderConfig,
    )

    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            deepstack_visual_indexes=[],
        ),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
    )
    original = Qwen3VLForConditionalGeneration(config).eval()
    with TemporaryDirectory() as path:
        original.save_pretrained(path)
        headless = Qwen3VLModel.from_pretrained(path, local_files_only=True).eval()
    for name, tensor in original.model.state_dict().items():
        assert torch.equal(tensor, headless.state_dict()[name])
    encoder = Qwen3VLEncoder.__new__(Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.config = Qwen3VLEncoderConfig(hidden_layer=2, dtype=torch.float32)
    encoder.model = headless
    encoder.processor = SimpleNamespace(
        create_mm_token_type_ids=lambda ids: [[0] * len(ids[0])]
    )
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = original.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.zeros_like(ids),
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states[2]
    actual = encoder({"token_ids": [1, 2, 3]})
    assert torch.equal(actual, expected)
