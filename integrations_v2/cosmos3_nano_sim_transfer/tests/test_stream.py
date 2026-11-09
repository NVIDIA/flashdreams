# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The streaming step API on the tiny stack of ``conftest.py``.

The stream is checked against the parts it wraps: driving the control
resize, the encoder, :class:`TransferRollout` and the decoder by hand gives
the same frames. Then the session contract a live host relies on: one frame,
then four per step, in and out; start, reset and a changed prompt, seed or
history; validation that leaves the session untouched; close.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_transfer.config import HistoryProfile, TransferModelConfig
from cosmos3_nano_sim_transfer.impl.control import resize_control, to_model_range
from cosmos3_nano_sim_transfer.impl.decode import to_uint8
from cosmos3_nano_sim_transfer.impl.prompt import HEIGHT, WIDTH
from cosmos3_nano_sim_transfer.impl.rollout import TransferRollout
from cosmos3_nano_sim_transfer.impl.stream import TransferStream, nominal_clip_frames

pytestmark = pytest.mark.ci_cpu

SEED = 20261012
PROMPT = "A taxi drives through a city."
TIMING_KEYS = {"encode_ms", "model_ms", "decode_ms", "to_host_ms", "step_ms"}

MakeStream = Callable[..., TransferStream]


def _controls(count: int, size: tuple[int, int] = (HEIGHT, WIDTH)) -> list[np.ndarray]:
    """Control frames per latent, ``[T, H, W, 3]`` uint8: 1, then 4 each."""
    rng = np.random.default_rng(3)
    palette = rng.integers(0, 256, size=(6, 3), dtype=np.uint8)
    out = []
    for k in range(count):
        labels = rng.integers(0, len(palette), size=(1 if k == 0 else 4, *size))
        out.append(palette[labels])
    return out


def _run(
    stream: TransferStream, controls: list[np.ndarray], **start
) -> list[np.ndarray]:
    stream.start(PROMPT, SEED, **start)
    return [stream.step(c) for c in controls]


def test_steps_take_one_then_four_frames_and_return_as_many(
    make_stream: MakeStream,
) -> None:
    stream = make_stream()
    assert (stream.width, stream.height) == (WIDTH, HEIGHT)
    assert stream.start(prompt=PROMPT, seed=SEED) == "new"
    for k, control in enumerate(_controls(4)):
        frames = stream.step(control)
        assert frames.dtype == np.uint8
        assert frames.shape == (1 if k == 0 else 4, HEIGHT, WIDTH, 3)
        timing = stream.last_timing
        assert set(timing) == TIMING_KEYS
        assert all(isinstance(v, float) and v >= 0 for v in timing.values())
        assert stream.latents == k + 1
    assert stream.cached_tokens > 0


def test_the_stream_is_the_parts_it_wraps(
    make_stream: MakeStream, tiny_network, tiny_model
) -> None:
    """Resize, encode, rollout and decode by hand give the stream's frames."""
    controls = _controls(5, size=(240, 416))  # resized to 832x480 per the hint
    stream = make_stream(hint="seg")
    ours = _run(stream, controls)

    by_hand_stream = make_stream()
    encoder, decoder = by_hand_stream.encoder, by_hand_stream.decoder
    assert encoder is not None and decoder is not None
    rollout = TransferRollout(
        tiny_network,
        tiny_model,
        history=HistoryProfile(3, 1),
        latent_hw=(4, 6),
        noise_dtype=torch.float32,
    )
    from cosmos3_nano_sim_transfer.impl import stream as stream_module

    ids = stream_module.prompt_token_ids(
        PROMPT,
        num_frames=nominal_clip_frames(30.0),
        fps=30.0,
        tokenizer_path="unused",
        max_tokens=None,
    )
    rollout.start(ids, seed=SEED)
    encoder.reset()
    decoder.reset()
    for control, got in zip(controls, ours, strict=True):
        pixels = torch.from_numpy(control).permute(0, 3, 1, 2).contiguous()
        latent = encoder.encode(to_model_range(resize_control(pixels, "seg")))
        frames = to_uint8(decoder.decode(rollout.step(latent)))
        np.testing.assert_array_equal(got, frames.permute(0, 2, 3, 1).numpy())


def test_tensor_steps_stay_on_the_device_in_tchw(make_stream: MakeStream) -> None:
    stream = make_stream()
    stream.start(PROMPT, SEED)
    controls = _controls(2)
    first = stream.step_tensor(torch.from_numpy(controls[0]))
    assert first.dtype == torch.uint8 and first.shape == (1, 3, HEIGHT, WIDTH)
    assert "to_host_ms" not in stream.last_timing
    second = stream.step_tensor(controls[1])
    assert second.shape == (4, 3, HEIGHT, WIDTH)

    host = make_stream()
    host_frames = _run(host, controls)
    np.testing.assert_array_equal(host_frames[1], second.permute(0, 2, 3, 1).numpy())


def test_start_reset_and_reseed(make_stream: MakeStream) -> None:
    controls = _controls(3)
    stream = make_stream()
    first = _run(stream, controls)
    assert stream.start(PROMPT, SEED) == "reset"  # same prompt: its cache is reused
    again = [stream.step(c) for c in controls]
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))

    assert stream.reset() == "reset"
    assert np.array_equal(stream.step(controls[0]), first[0])

    stream.start(PROMPT, SEED + 1)
    assert not np.array_equal(stream.step(controls[0]), first[0])
    assert stream.start("Another prompt entirely.", SEED) == "new"

    fresh = make_stream()
    assert all(
        np.array_equal(a, b) for a, b in zip(first, _run(fresh, controls), strict=True)
    )


def test_stream_forwards_explicit_emphasis_and_active_hint(
    make_stream: MakeStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos3_nano_sim_transfer.impl import stream as stream_module

    stream = make_stream(
        hint="depth", emphasize_control_in_prompt=True, prompt_aspect_ratio="16,9"
    )
    seen = []

    def ids(caption: str, **kwargs) -> torch.Tensor:
        seen.append(kwargs)
        return torch.tensor([1, 2, 3])

    monkeypatch.setattr(stream_module, "prompt_token_ids", ids)
    stream.start(PROMPT, SEED)
    assert seen[0]["hint"] == "depth"
    assert seen[0]["emphasize_control_in_prompt"] is True
    assert seen[0]["aspect_ratio"] == "16,9"


def test_start_resets_the_encoder_and_decoder(make_stream: MakeStream) -> None:
    stream = make_stream()
    controls = _controls(2)
    _run(stream, controls)
    _run(stream, controls)  # would fail on a 1-frame step mid-clip without resets
    encoder, decoder = stream.encoder, stream.decoder
    assert getattr(encoder, "resets", None) == 2
    assert getattr(decoder, "resets", None) == 2


def test_history_per_session(make_stream: MakeStream) -> None:
    controls = _controls(6)
    stream = make_stream()
    assert stream.history == HistoryProfile(3, 1)
    windowed = _run(stream, controls)
    before = stream.rollout
    assert before is not None
    other = _run(stream, controls, history="4/0")
    assert stream.history == HistoryProfile(4, 0)
    after = stream.rollout
    assert after is not None and after.cache is not before.cache
    assert not np.array_equal(windowed[-1], other[-1])
    assert all(
        np.array_equal(a, b)
        for a, b in zip(other, _run(make_stream(history="4/0"), controls), strict=True)
    )
    assert stream.start(PROMPT, SEED, history="long-rollout") == "new"
    assert stream.history == HistoryProfile(30, 3)


def test_fewer_steps_are_the_rollout_on_the_shorter_schedule(
    make_stream: MakeStream, tiny_model: TransferModelConfig
) -> None:
    controls = _controls(4)
    stream = make_stream(t_list=(1.0, 0.625))
    assert stream.t_list == stream.model.t_list == (1.0, 0.625)
    assert make_stream().t_list == tiny_model.t_list
    two = _run(stream, controls)
    same = _run(make_stream(model=tiny_model.with_schedule(steps=2)), controls)
    assert all(np.array_equal(a, b) for a, b in zip(two, same, strict=True))
    four = _run(make_stream(), controls)
    assert not np.array_equal(two[-1], four[-1])
    for bad in ((0.9375, 0.625), (1.0, 1.0), ()):
        with pytest.raises(ValueError):
            make_stream(t_list=bad)


def test_load_checks_the_schedule_before_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``steps`` and ``t_list`` are resolved from ``checkpoint.json`` before the network loads."""
    import cosmos_interactive.checkpoint_load as checkpoint_load

    (tmp_path / "checkpoint.json").write_text(
        (Path(__file__).parent / "data" / "checkpoint_metadata.json").read_text()
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.net.vae2llm.weight": "model.safetensors"}}'
    )
    (tmp_path / "model.safetensors").touch()

    def no_load(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("the network was loaded")

    monkeypatch.setattr(checkpoint_load, "load_exported_checkpoint", no_load)
    with pytest.raises(ValueError, match="steps must be in"):
        TransferStream.load(tmp_path, device="cpu", steps=5)
    with pytest.raises(ValueError, match="not both"):
        TransferStream.load(tmp_path, device="cpu", steps=2, t_list=(1.0, 0.625))
    with pytest.raises(ValueError, match="must start"):
        TransferStream.load(tmp_path, device="cpu", t_list=(0.5,))


def test_the_prompt_states_the_nominal_or_given_clip_length(
    make_stream: MakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos3_nano_sim_transfer.impl import stream as stream_module

    seen: list[int] = []
    original = stream_module.prompt_token_ids

    def spy(caption, *, num_frames, **kwargs):  # noqa: ANN001, ANN202
        seen.append(num_frames)
        return original(caption, num_frames=num_frames, **kwargs)

    monkeypatch.setattr(stream_module, "prompt_token_ids", spy)
    stream = make_stream()
    stream.start(PROMPT, SEED)
    stream.start(PROMPT, SEED, num_frames=201)
    stream.reset()
    assert seen == [601, 201, 201]
    assert make_stream(fps=20.0).nominal_frames == 401
    assert make_stream(nominal_frames=97).nominal_frames == 97
    assert nominal_clip_frames(30.0) == 601 and nominal_clip_frames(15.0) == 301
    with pytest.raises(ValueError):
        nominal_clip_frames(0.0)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, HEIGHT, WIDTH, 3), np.uint8),  # four frames for the first latent
        np.zeros((1, HEIGHT, WIDTH, 3), np.float32),
        np.zeros((1, 3, HEIGHT, WIDTH), np.uint8),  # channels first
        np.zeros((HEIGHT, WIDTH, 3), np.uint8),
    ],
    ids=["count", "dtype", "layout", "rank"],
)
def test_invalid_controls_leave_the_session_untouched(
    make_stream: MakeStream, bad: np.ndarray
) -> None:
    stream = make_stream()
    stream.start(PROMPT, SEED)
    with pytest.raises(ValueError):
        stream.step(bad)
    assert stream.latents == 0
    expected = _run(make_stream(), _controls(1))[0]
    assert np.array_equal(stream.step(_controls(1)[0]), expected)


def test_session_guards(make_stream: MakeStream) -> None:
    stream = make_stream()
    control = _controls(1)[0]
    with pytest.raises(RuntimeError):
        stream.step(control)
    with pytest.raises(RuntimeError):
        stream.reset()
    stream.start(PROMPT, SEED)
    stream.step(control, last=True)
    with pytest.raises(RuntimeError):  # the session ended with its last latent
        stream.step(_controls(2)[1])
    stream.reset()
    stream.step(control)
    with pytest.raises(ValueError):
        make_stream(hint="normals")
    with pytest.raises(ValueError):
        make_stream(fps=0.0)
    stream.close()
    with pytest.raises(RuntimeError):
        stream.start(PROMPT, SEED)
    with pytest.raises(RuntimeError):
        stream.step(control)


def test_hint_must_be_one_the_checkpoint_was_trained_on(
    make_stream: MakeStream, tiny_model: TransferModelConfig
) -> None:
    make_stream(hint="depth").close()
    seg_only = dataclasses.replace(tiny_model, hints=("seg",))
    make_stream(model=seg_only, hint="seg").close()
    with pytest.raises(ValueError, match="trained on"):
        make_stream(model=seg_only, hint="depth")
