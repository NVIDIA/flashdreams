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

"""AgiBot generic diffusion lifecycle against a tiny CPU network.

These exercise the real transformer and cache through ``DiffusionModel`` so the
model mapping, cache lifecycle, and rollout math are covered end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn
from cosmos3_nano_sim_bimanual import action_format as _agibot
from cosmos3_nano_sim_bimanual.encoder import HandActionInput
from cosmos3_nano_sim_bimanual.spec import (
    Cosmos3InteractiveSpec,
    LLMSpec,
    VFMSpec,
    VisionSpec,
)
from cosmos3_nano_sim_bimanual.transformer import (
    HandTransformer,
    HandTransformerConfig,
)
from cosmos3_nano_sim_bimanual.transformer.cache import HandTransformerCache
from cosmos3_nano_sim_bimanual.transformer.config import CacheMode, CacheRefreshMode
from cosmos3_nano_sim_bimanual.transformer.network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
)

from flashdreams.infra.compile import CompileMode
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig

pytestmark = pytest.mark.ci_cpu

_GRID = (2, 2)
_TCF = 4
_CHUNK = 2


class _Session:
    """Test convenience around the real generic model; production has no adapter."""

    def __init__(
        self,
        model: DiffusionModel[HandTransformerCache],
        cache: HandTransformerCache,
    ) -> None:
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "transformer", model.transformer)
        object.__setattr__(self, "cache_object", cache)

    def __getattr__(self, name: str):
        return getattr(self.cache_object, name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"model", "transformer", "cache_object"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.cache_object, name, value)

    def step(
        self, actions: torch.Tensor | HandActionInput, *, profile: bool = False
    ) -> list[torch.Tensor]:
        self.cache_object.profile = profile
        clean, final_state = self.model.generate(
            self.cache_object._steps,
            self.cache_object,
            input=actions,
        )
        self.model.finalize(final_state)
        return [clean[0, :, index : index + 1] for index in range(clean.shape[2])]

    def reset_rollout(self) -> None:
        self.cache_object.reset_rollout()

    def prewarm_rollout(
        self, steps: int, *, profile: bool = False
    ) -> list[dict[str, float]]:
        timings = []
        for _ in range(steps):
            self.step(_actions(_CHUNK), profile=profile)
            timings.append(dict(self.last_timing))
        self.reset_rollout()
        return timings


def _net() -> tuple[CosmosMoTNetwork, Cosmos3InteractiveSpec]:
    spec = Cosmos3InteractiveSpec(
        llm=LLMSpec(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=100,
            mrope_section=(4, 2, 2),
            rope_theta=1_000_000.0,
        ),
        vision=VisionSpec(),
        vfm=VFMSpec(vae_latent_channels=8, latent_patch_size=2),
    )
    return (
        CosmosMoTNetworkConfig(
            spec=spec,
            dtype=torch.float32,
            include_vision_tower=False,
            action_gen=True,
            action_dim=64,
            num_embodiment_domains=_agibot.AGIBOT_DOMAIN_ID + 1,
        ).setup(),
        spec,
    )


def _session(
    *,
    text: bool = True,
    window_frames: int = 0,
    native_gqa_attention: bool = False,
    sdpa_backend: str = "default",
    cache_mode: CacheMode = "cat",
    cache_refresh_mode: CacheRefreshMode = "chunk",
    overlap_cache_update: bool = False,
    compile_denoise: bool = False,
    compile_denoise_mode: CompileMode = "default",
    fixed_shape_cache_fill: bool = False,
    valid_length_cache_attention: bool = False,
    cuda_graph_denoise: bool = False,
    shared: tuple[CosmosMoTNetwork, Cosmos3InteractiveSpec] | None = None,
) -> _Session:
    net, spec = shared if shared is not None else _net()
    generator = torch.Generator().manual_seed(0)
    hw = _GRID[0] * _GRID[1]
    init = torch.randn(hw, spec.vfm.patch_latent_dim, generator=generator)
    text_ids = (
        torch.randint(0, spec.llm.vocab_size, (3,), generator=generator)
        if text
        else None
    )
    transformer_config = HandTransformerConfig(
        device="cpu",
        chunk_size=_CHUNK,
        window_frames=window_frames,
        native_gqa_attention=native_gqa_attention,
        sdpa_backend=sdpa_backend,
        cache_mode=cache_mode,
        cache_refresh_mode=cache_refresh_mode,
        overlap_cache_update=overlap_cache_update,
        compile_denoise=compile_denoise,
        compile_denoise_mode=compile_denoise_mode,
        fixed_shape_cache_fill=fixed_shape_cache_fill,
        valid_length_cache_attention=valid_length_cache_attention,
        cuda_graph_denoise=cuda_graph_denoise,
    )
    transformer = HandTransformer(
        transformer_config,
        network=net,
    )
    cache = HandTransformerCache(
        net,
        scene=SimpleNamespace(name="test"),
        text_ids=text_ids,
        init_latent=init,
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
        window_frames=window_frames,
        native_gqa_attention=native_gqa_attention,
        sdpa_backend=sdpa_backend,
        cache_mode=cache_mode,
        cache_refresh_mode=cache_refresh_mode,
        overlap_cache_update=overlap_cache_update,
        compile_denoise=compile_denoise,
        compile_denoise_mode=compile_denoise_mode,
        fixed_shape_cache_fill=fixed_shape_cache_fill,
        valid_length_cache_attention=valid_length_cache_attention,
        cuda_graph_denoise=cuda_graph_denoise,
    )
    config = DiffusionModelConfig(
        transformer=transformer_config,
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=4,
            shift=5.0,
            denoising_timesteps=[1000, 750, 500, 250],
        ),
        seed=0,
    )
    model = DiffusionModel.__new__(DiffusionModel)
    nn.Module.__init__(model)
    model.config = config
    model.transformer = transformer
    model.scheduler = config.scheduler.setup()
    model._rng = None
    return _Session(model, cache)


def _actions(frames: int) -> torch.Tensor:
    raw = _agibot.identity_raw_action(frames * _TCF)
    return torch.from_numpy(_agibot.normalize_and_pad(raw))


def test_session_constructs_and_seeds_frame_zero() -> None:
    session = _session()
    # Frame 0 is the pinned sink; generation starts at 1.
    assert session._abs_frame == 1
    assert session._per_frame_tokens == _GRID[0] * _GRID[1] + _TCF
    assert session.cache, "the KV cache was never populated"


def test_frozen_tiny_rollout_through_cache_eviction(monkeypatch) -> None:
    torch.manual_seed(17)
    session = _session(window_frames=4, cache_mode="block")
    # Seeded BF16 CPU noise differs across PyTorch versions. Freeze this
    # fixture's inputs with FP32 draws rounded to the requested dtype.
    randn = torch.randn

    def fixture_noise(size, *, device, dtype, generator):
        return randn(size, device=device, dtype=torch.float32, generator=generator).to(
            dtype
        )

    monkeypatch.setattr(torch, "randn", fixture_noise)
    # GA stationary unified actions, 30 FPS conditioning, four-step SDE.
    expected = [
        [
            0.0990821272,
            0.8152567148,
            0.8270251751,
            0.6233447194,
            1.2947176695,
            0.6764248610,
        ],
        [
            0.0070223534,
            0.7851077914,
            -1.5744485855,
            -0.0905759335,
            0.0952403396,
            0.0048038661,
        ],
        [
            0.0188023634,
            0.8335111737,
            0.0833756775,
            0.6577188969,
            -0.8082447648,
            -0.9016210437,
        ],
        [
            -0.0003055613,
            0.8294845223,
            -1.2086137533,
            0.0808462799,
            -1.0328346491,
            -1.2227172852,
        ],
    ]
    with torch.inference_mode():
        for reference in expected:
            output = torch.cat(session.step(_actions(_CHUNK)), dim=1)
            actual = torch.cat(
                (output.mean()[None], output.std()[None], output.flatten()[:4])
            )
            torch.testing.assert_close(
                actual, torch.tensor(reference), rtol=2e-5, atol=2e-5
            )


def test_session_constructs_without_text_conditioning() -> None:
    session = _session(text=False)
    assert session.n_text == 0
    assert session._abs_frame == 1


def test_step_returns_one_latent_frame_per_action_group() -> None:
    session = _session()
    with torch.no_grad():
        frames = session.step(_actions(_CHUNK))
    assert len(frames) == _CHUNK
    for frame in frames:
        assert frame.shape == (8, 1, _GRID[0] * 2, _GRID[1] * 2)
        assert torch.isfinite(frame).all()
    assert session._abs_frame == 1 + _CHUNK


def test_prewarm_rollout_resets_to_the_first_visible_chunk() -> None:
    shared = _net()
    fresh = _session(shared=shared, window_frames=2 * _CHUNK, cache_mode="block")
    prewarmed = _session(shared=shared, window_frames=2 * _CHUNK, cache_mode="block")

    timings = prewarmed.prewarm_rollout(4, profile=True)

    assert len(timings) == 4
    assert prewarmed._abs_frame == fresh._abs_frame == 1
    assert prewarmed._steps == fresh._steps == 0
    assert prewarmed._cached_tokens() == fresh._cached_tokens()
    with torch.no_grad():
        expected = fresh.step(_actions(_CHUNK))
        actual = prewarmed.step(_actions(_CHUNK))
    for expected_frame, actual_frame in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_frame, expected_frame, rtol=1e-3, atol=1e-4)


def test_sequential_rng_restarts_for_each_rollout() -> None:
    session = _session()
    cache = session.cache_object
    generator = torch.Generator(device=session.transformer.device).manual_seed(47)
    kwargs = {
        "latent_shape": session.transformer.latent_shape,
        "rng": generator,
        "cache": cache,
        "input": _actions(_CHUNK),
    }

    first = session.transformer.initial_noise(**kwargs)
    cache._steps = 1
    second = session.transformer.initial_noise(**kwargs)
    cache.reset_rollout()
    restarted = session.transformer.initial_noise(**kwargs)

    assert not torch.equal(second, first)
    torch.testing.assert_close(restarted, first, rtol=0, atol=0)


def test_custom_tcf_drives_vision_mrope(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos3_nano_sim_bimanual.transformer.cache as cache_module

    seen: list[int] = []
    vae_mrope_ids = cache_module.vae_mrope_ids

    def recording_vae_mrope_ids(*args, **kwargs):
        seen.append(kwargs["temporal_compression_factor"])
        return vae_mrope_ids(*args, **kwargs)

    monkeypatch.setattr(cache_module, "vae_mrope_ids", recording_vae_mrope_ids)
    net, spec = _net()
    custom_tcf = 2
    HandTransformerCache(
        net,
        scene=SimpleNamespace(name="custom-tcf"),
        text_ids=None,
        init_latent=torch.zeros(_GRID[0] * _GRID[1], spec.vfm.patch_latent_dim),
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=custom_tcf,
    )

    assert seen
    assert set(seen) == {custom_tcf}


def test_consecutive_steps_advance_the_absolute_frame() -> None:
    session = _session()
    with torch.no_grad():
        session.step(_actions(_CHUNK))
        second = session.step(_actions(_CHUNK))
    assert len(second) == _CHUNK
    assert session._abs_frame == 1 + 2 * _CHUNK


def test_a_partial_chunk_is_accepted() -> None:
    session = _session()
    with torch.no_grad():
        frames = session.step(_actions(1))
    assert len(frames) == 1
    assert session._abs_frame == 2


def test_step_rejects_a_ragged_action_chunk() -> None:
    session = _session()
    with pytest.raises(ValueError, match="rows of 64"):
        session.step(torch.zeros(_TCF + 1, _agibot.MAX_ACTION_DIM))


def test_step_rejects_more_frames_than_the_chunk_size() -> None:
    session = _session()
    with pytest.raises(ValueError, match="maximum is"):
        session.step(_actions(_CHUNK + 1))


def test_window_frames_bounds_the_cache() -> None:
    unbounded = _session(window_frames=0)
    bounded = _session(window_frames=2)
    with torch.no_grad():
        for _ in range(3):
            unbounded.step(_actions(_CHUNK))
            bounded.step(_actions(_CHUNK))
    grew = unbounded._cat_cache()[0]["k"].shape[2]
    capped = bounded._cat_cache()[0]["k"].shape[2]
    assert capped < grew, "the window did not prune the cache"
    # Text plus the frame-0 sink are always retained.
    assert capped >= unbounded.n_text + unbounded._per_frame_tokens


def test_a_bounded_window_stops_the_context_from_growing() -> None:
    """The property that matters: attention length must reach a ceiling.

    An unbounded cache makes every step slower than the last, which is what made
    live chunk times climb through a session.
    """
    session = _session(window_frames=2)
    lengths = []
    with torch.no_grad():
        for _ in range(6):
            session.step(_actions(_CHUNK))
            lengths.append(session._cat_cache()[0]["k"].shape[2])
    assert lengths[-1] == lengths[-2], f"context still growing: {lengths}"
    expected = session.n_text + session._per_frame_tokens * (1 + session._window_frames)
    assert lengths[-1] == expected


def test_an_unbounded_window_keeps_growing() -> None:
    session = _session(window_frames=0)
    lengths = []
    with torch.no_grad():
        for _ in range(4):
            session.step(_actions(_CHUNK))
            lengths.append(session._cat_cache()[0]["k"].shape[2])
    assert lengths == sorted(lengths) and lengths[-1] > lengths[0]


def test_the_default_window_is_bounded() -> None:
    from cosmos3_nano_sim_bimanual.transformer.cache import DEFAULT_WINDOW_FRAMES

    assert DEFAULT_WINDOW_FRAMES > 0
    net, spec = _net()
    session = HandTransformerCache(
        net,
        scene=SimpleNamespace(name="test"),
        text_ids=None,
        init_latent=torch.zeros(_GRID[0] * _GRID[1], spec.vfm.patch_latent_dim),
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
    )
    assert session._window_frames == DEFAULT_WINDOW_FRAMES


def test_profile_records_per_stage_timings() -> None:
    session = _session()
    with torch.no_grad():
        session.step(_actions(_CHUNK))
    assert session.last_timing == {}, "timings recorded without asking"

    with torch.no_grad():
        session.step(_actions(_CHUNK), profile=True)
    timing = session.last_timing
    assert {
        "model_step_s",
        "setup_s",
        "denoise_s",
        "cache_seed_prune_s",
        "cache_update_submit_s",
        "cache_update_wait_s",
        "output_s",
        "cached_tokens",
    } <= set(timing)
    # The stages must partition the whole step, which is what makes the breakdown
    # usable for attributing a slow chunk rather than merely suggestive.
    stages = sum(
        timing[key]
        for key in ("setup_s", "denoise_s", "cache_seed_prune_s", "output_s")
    )
    assert stages == pytest.approx(timing["model_step_s"], rel=0.02, abs=1e-4)
    assert timing["cached_tokens"] > 0


def test_native_gqa_is_honoured() -> None:
    session = _session(native_gqa_attention=True)
    assert session.native_gqa_attention is True
    with torch.no_grad():
        frames = session.step(_actions(_CHUNK))
    assert len(frames) == _CHUNK


def test_an_unknown_sdpa_backend_is_rejected() -> None:
    with pytest.raises(ValueError):
        _session(sdpa_backend="nonsense")


def test_a_pinned_sdpa_backend_still_steps() -> None:
    session = _session(sdpa_backend="math")
    assert session.sdpa_backend == "math"
    with torch.no_grad():
        assert len(session.step(_actions(_CHUNK))) == _CHUNK


def _count_forwards(session: _Session, monkeypatch) -> list[int]:
    """Record the token count of every transformer pass one step makes."""
    seen: list[int] = []
    layers_step = session.net._layers_step
    gen_step = [layer.gen_step for layer in session._layer_list()]

    def counting_layers_step(h, *args, **kwargs):
        seen.append(int(h.shape[1]))
        return layers_step(h, *args, **kwargs)

    monkeypatch.setattr(session.net, "_layers_step", counting_layers_step)
    for index, layer in enumerate(session._layer_list()):
        # The stream path runs the layers itself, so count its first layer only.
        def counting_gen_step(h, *args, _index=index, _inner=gen_step[index], **kwargs):
            if _index == 0:
                seen.append(int(h.shape[1]))
            return _inner(h, *args, **kwargs)

        monkeypatch.setattr(layer, "gen_step", counting_gen_step)
    return seen


def test_a_step_costs_one_forward_per_denoise_plus_one_to_seed(monkeypatch) -> None:
    """Seeding a chunk per frame silently triples the non-denoise cost.

    A four-step schedule over a four-frame chunk should be four denoise passes
    and one batched seed, not four and four: the extra passes are pure overhead
    and were the difference between 19 and 26 generated fps.
    """
    session = _session(window_frames=4 * _CHUNK)
    seen = _count_forwards(session, monkeypatch)
    with torch.no_grad():
        session.step(_actions(_CHUNK))
    assert len(seen) == session.model.scheduler.config.num_inference_steps + 1, (
        f"forward sizes: {seen}"
    )
    # Every pass covers the whole chunk, so none of them is a per-frame seed.
    assert set(seen) == {_CHUNK * session._per_frame_tokens}


def test_clean_cache_refresh_precedes_generic_finalize(monkeypatch) -> None:
    session = _session(window_frames=4 * _CHUNK)
    refreshes = 0
    refresh = session._seed_chunk_cat

    def recording_refresh(*args, **kwargs):
        nonlocal refreshes
        refreshes += 1
        return refresh(*args, **kwargs)

    monkeypatch.setattr(session.cache_object, "_seed_chunk_cat", recording_refresh)
    clean, final_state = session.model.generate(
        0,
        session.cache_object,
        input=_actions(_CHUNK),
    )

    assert clean.shape[2] == _CHUNK
    assert refreshes == 1
    session.model.finalize(final_state)
    assert refreshes == 1


def test_a_batched_seed_matches_seeding_frame_by_frame(monkeypatch) -> None:
    """Batching is an equivalence, not an approximation.

    One causal-across-frames pass shows each frame exactly the frames before it,
    which is what a sequential seed through a growing cache does.
    """
    shared = _net()
    batched = _steps(_session(shared=shared, window_frames=4 * _CHUNK), 2)

    per_frame = _session(shared=shared, window_frames=4 * _CHUNK)

    def seed_each_frame(start, clean_vision, action, n_frames, domain_ids):
        for index in range(n_frames):
            per_frame._seed_frame_cat(
                start + index,
                clean_vision[index * per_frame.hw : (index + 1) * per_frame.hw],
                action[index * per_frame.tcf : (index + 1) * per_frame.tcf],
                domain_ids=domain_ids[
                    index * per_frame.tcf : (index + 1) * per_frame.tcf
                ],
            )

    monkeypatch.setattr(per_frame, "_seed_chunk_cat", seed_each_frame)
    sequential = _steps(per_frame, 2)
    for index, (expected, actual) in enumerate(zip(sequential, batched, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


def test_the_window_rounds_up_to_whole_chunks() -> None:
    """Both cache modes must span the same history to be comparable.

    The ring buffer can only evict a whole block, so the window is quantized to
    the chunk for every mode rather than only for the one that needs it.
    """
    assert _session(window_frames=_CHUNK + 1)._window_frames == 2 * _CHUNK
    assert _session(window_frames=_CHUNK)._window_frames == _CHUNK


def _steps(session: _Session, count: int) -> list[torch.Tensor]:
    with torch.no_grad():
        return [frame for _ in range(count) for frame in session.step(_actions(_CHUNK))]


def test_block_mode_matches_cat_mode() -> None:
    """The ring buffer is an allocation strategy, not a change of math.

    Both modes are run against one set of weights with a window wide enough that
    neither evicts, so any difference is a bug in the block plumbing rather than
    an expected consequence of a shorter history.
    """
    shared = _net()
    window = 4 * _CHUNK
    cat = _steps(_session(shared=shared, window_frames=window, cache_mode="cat"), 2)
    block = _steps(_session(shared=shared, window_frames=window, cache_mode="block"), 2)
    assert len(cat) == len(block)
    for index, (expected, actual) in enumerate(zip(cat, block, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


def test_framewise_refresh_matches_cat_mode() -> None:
    """Framewise refresh exposes real history instead of masking one forward.

    Frames are refreshed against a prefix that is already clean, so the result
    should still track the sequential cat path.
    """
    shared = _net()
    window = 4 * _CHUNK
    cat = _steps(_session(shared=shared, window_frames=window, cache_mode="cat"), 2)
    framewise = _steps(
        _session(
            shared=shared,
            window_frames=window,
            cache_mode="block",
            cache_refresh_mode="framewise",
        ),
        2,
    )
    for index, (expected, actual) in enumerate(zip(cat, framewise, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


def test_block_mode_holds_its_context_flat_past_eviction() -> None:
    session = _session(window_frames=_CHUNK, cache_mode="block")
    lengths = []
    with torch.no_grad():
        for _ in range(6):
            session.step(_actions(_CHUNK))
            lengths.append(session._cached_tokens())
    ceiling = session.n_text + session._per_frame_tokens * (1 + session._window_frames)
    assert lengths[-1] == ceiling, f"{lengths} should settle at {ceiling}"
    assert lengths[-1] == lengths[-2]


def test_block_mode_keeps_the_start_image_pinned() -> None:
    """Frame 0 is the I2V conditioning and must survive every eviction."""
    session = _session(window_frames=_CHUNK, cache_mode="block")
    assert session.stream_cache is not None
    sink = session._per_frame_tokens
    before = session.stream_cache.gen[0]._k[:, :, :sink].clone()
    with torch.no_grad():
        for _ in range(6):
            session.step(_actions(_CHUNK))
    after = session.stream_cache.gen[0]._k[:, :, :sink]
    torch.testing.assert_close(after, before)


def test_block_mode_zeroes_the_null_action_values() -> None:
    """The start image conditions geometry without implying motion.

    The cat path gets this from ``zero_value_mask``; the block path has to zero
    the values itself when writing the sink, and forgetting to would let every
    later frame read a phantom action off frame 0.
    """
    session = _session(window_frames=_CHUNK, cache_mode="block")
    assert session.stream_cache is not None
    values = session.stream_cache.gen[0]._v
    action = values[:, :, : session.tcf]
    vision = values[:, :, session.tcf : session._per_frame_tokens]
    assert torch.count_nonzero(action) == 0
    assert torch.count_nonzero(vision) > 0


def test_block_mode_requires_a_bounded_window() -> None:
    with pytest.raises(ValueError, match="window_frames"):
        _session(window_frames=0, cache_mode="block")


def test_block_mode_rejects_a_partial_chunk() -> None:
    session = _session(window_frames=_CHUNK, cache_mode="block")
    with pytest.raises(ValueError, match="full"), torch.no_grad():
        session.step(_actions(_CHUNK - 1))


def test_cat_mode_still_accepts_a_partial_chunk() -> None:
    session = _session(window_frames=_CHUNK, cache_mode="cat")
    with torch.no_grad():
        assert len(session.step(_actions(_CHUNK - 1))) == _CHUNK - 1


def test_framewise_refresh_requires_block_mode() -> None:
    with pytest.raises(ValueError, match="framewise"):
        _session(cache_mode="cat", cache_refresh_mode="framewise")


def test_an_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache_mode"):
        _session(cache_mode="nonsense")  # ty: ignore[invalid-argument-type]


def test_an_unknown_refresh_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache_refresh_mode"):
        _session(cache_refresh_mode="nonsense")  # ty: ignore[invalid-argument-type]


def test_transformer_config_rejects_bad_values_before_loading_a_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad command line should cost seconds, not a checkpoint load.

    The integration config shares the session's rules, so the app can refuse to
    start before it fetches weights.
    """
    loaded = False

    def load_network(self: HandTransformer) -> CosmosMoTNetwork:
        nonlocal loaded
        loaded = True
        return _net()[0]

    monkeypatch.setattr(HandTransformer, "_load_network", load_network)
    invalid = [
        (
            HandTransformerConfig(device="cpu", cache_mode="nonsense"),  # ty: ignore[invalid-argument-type]
            "cache_mode",
        ),
        (
            HandTransformerConfig(device="cpu", cache_refresh_mode="framewise"),
            "framewise",
        ),
        (
            HandTransformerConfig(device="cpu", overlap_cache_update=True),
            "overlap_cache_update",
        ),
        (HandTransformerConfig(device="cpu", compile_denoise=True), "compile_denoise"),
        (
            HandTransformerConfig(device="cpu", cache_mode="block", window_frames=0),
            "window_frames",
        ),
        (
            HandTransformerConfig(
                device="cpu",
                cache_mode="block",
                compile_denoise=True,
                compile_denoise_mode="nonsense",  # ty: ignore[invalid-argument-type]
            ),
            "compile denoise mode",
        ),
        (HandTransformerConfig(device="cpu", model_fps=0.0), "model_fps"),
        (HandTransformerConfig(device="cpu", model_fps=float("nan")), "model_fps"),
    ]
    for config, message in invalid:
        with pytest.raises(ValueError, match=message):
            HandTransformer(config)
    assert not loaded


@pytest.mark.parametrize(
    ("model_fps", "expected_fps"),
    [(None, 30.0), (36.0, 36.0)],
)
def test_transformer_plumbs_temporal_fps(
    monkeypatch, model_fps: float | None, expected_fps: float
) -> None:
    import cosmos3_nano_sim_bimanual.transformer as transformer_module

    captured: dict[str, object] = {}
    expected_cache = object()

    def fake_cache(net, **kwargs):
        captured["net"] = net
        captured.update(kwargs)
        return expected_cache

    monkeypatch.setattr(transformer_module, "HandTransformerCache", fake_cache)
    transformer = HandTransformer(
        HandTransformerConfig(
            device="cpu",
            chunk_size=_CHUNK,
            model_fps=model_fps,
            window_frames=8,
        ),
        network=cast(CosmosMoTNetwork, nn.Linear(1, 1)),
    )
    scene = SimpleNamespace(
        text_ids=torch.zeros(3, dtype=torch.long),
        init_latent=torch.zeros(4, 8),
        fps=30.0,
    )

    cache = transformer._new_cache(scene)

    assert cache is expected_cache
    assert captured["fps"] == expected_fps
    assert scene.fps == 30.0


def test_the_ring_buffer_keeps_a_stable_address() -> None:
    """A captured CUDA graph holds raw pointers into this buffer.

    Eviction has to shift data inside the existing allocation rather than
    reallocate, or a replayed graph would read freed memory.
    """
    session = _session(window_frames=_CHUNK, cache_mode="block")
    assert session.stream_cache is not None
    pointers = [
        (gen._k.data_ptr(), gen._v.data_ptr()) for gen in session.stream_cache.gen
    ]
    with torch.no_grad():
        for _ in range(6):
            session.step(_actions(_CHUNK))
    assert [
        (gen._k.data_ptr(), gen._v.data_ptr()) for gen in session.stream_cache.gen
    ] == pointers


def test_overlap_requires_block_mode() -> None:
    with pytest.raises(ValueError, match="overlap_cache_update"):
        _session(cache_mode="cat", overlap_cache_update=True)


def test_compile_requires_block_mode() -> None:
    with pytest.raises(ValueError, match="compile_denoise"):
        _session(cache_mode="cat", compile_denoise=True)


def test_fixed_shape_fill_requires_compile() -> None:
    with pytest.raises(ValueError, match="fixed_shape_cache_fill"):
        _session(
            window_frames=_CHUNK,
            cache_mode="block",
            fixed_shape_cache_fill=True,
        )


def test_valid_length_attention_requires_compile() -> None:
    with pytest.raises(ValueError, match="valid_length_cache_attention"):
        _session(
            window_frames=_CHUNK,
            cache_mode="block",
            valid_length_cache_attention=True,
        )


def test_valid_length_attention_rejects_masked_fixed_fill() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _session(
            window_frames=_CHUNK,
            cache_mode="block",
            compile_denoise=True,
            fixed_shape_cache_fill=True,
            valid_length_cache_attention=True,
        )


def test_cuda_graphs_require_compile() -> None:
    with pytest.raises(ValueError, match="compile_denoise"):
        _session(window_frames=_CHUNK, cache_mode="block", cuda_graph_denoise=True)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only device")
def test_cuda_graphs_require_a_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        _session(
            window_frames=_CHUNK,
            cache_mode="block",
            compile_denoise=True,
            cuda_graph_denoise=True,
        )


def test_an_unknown_compile_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="compile denoise mode"):
        _session(
            window_frames=_CHUNK,
            cache_mode="block",
            compile_denoise=True,
            compile_denoise_mode="nonsense",  # ty: ignore[invalid-argument-type]
        )


def test_compile_uses_dynamic_fill_then_static_steady_state(monkeypatch) -> None:
    """Growing history uses one symbolic graph before static graph takeover."""
    compiled: list[tuple[str | None, bool | None]] = []
    executed: list[bool | None] = []

    def fake_compile(fn, *, mode=None, dynamic=None, **_kwargs):
        compiled.append((mode, dynamic))

        def wrapped(*args, **kwargs):
            executed.append(dynamic)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    window = 3 * _CHUNK
    session = _session(window_frames=window, cache_mode="block", compile_denoise=True)
    assert compiled == [(None, False), (None, True)]
    assert session._compile_start_step == window // _CHUNK

    with torch.no_grad():
        session.step(_actions(_CHUNK), profile=True)
    assert executed == [True] * session.model.scheduler.config.num_inference_steps
    assert session.last_timing["compile_denoise_active"] == 1.0
    assert session.last_timing["compile_denoise_fill_active"] == 1.0
    assert session.last_timing["compile_denoise_static_active"] == 0.0

    with torch.no_grad():
        while session._steps < session._compile_start_step:
            session.step(_actions(_CHUNK), profile=True)
    assert session.last_timing["compile_denoise_fill_active"] == 1.0

    executed.clear()
    with torch.no_grad():
        session.step(_actions(_CHUNK), profile=True)
    assert executed == [False] * session.model.scheduler.config.num_inference_steps
    assert session.last_timing["compile_denoise_fill_active"] == 0.0
    assert session.last_timing["compile_denoise_static_active"] == 1.0


def test_compile_forwards_the_selected_mode_to_static_paths(monkeypatch) -> None:
    compiled: list[tuple[str | None, bool | None]] = []

    def fake_compile(fn, *, mode=None, dynamic=None, **_kwargs):
        compiled.append((mode, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    _session(
        window_frames=3 * _CHUNK,
        cache_mode="block",
        compile_denoise=True,
        compile_denoise_mode="reduce-overhead",
    )

    assert compiled == [("reduce-overhead", False), (None, True)]


def test_fixed_shape_fill_uses_a_static_callable_from_the_first_step(
    monkeypatch,
) -> None:
    """The opt-in fill path keeps one static shape until steady-state takeover."""
    compiled: list[tuple[str | None, bool | None]] = []

    def fake_compile(fn, *, mode=None, dynamic=None, **_kwargs):
        compiled.append((mode, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    session = _session(
        window_frames=3 * _CHUNK,
        cache_mode="block",
        compile_denoise=True,
        fixed_shape_cache_fill=True,
    )

    assert compiled == [(None, False), (None, False)]
    with torch.no_grad():
        session.step(_actions(_CHUNK), profile=True)
    assert session.last_timing["compile_denoise_fill_active"] == 1.0
    assert session.last_timing["compile_denoise_masked_fill_active"] == 1.0

    with torch.no_grad():
        while session._steps < session._compile_start_step:
            session.step(_actions(_CHUNK), profile=True)
        session.step(_actions(_CHUNK), profile=True)
    assert session.last_timing["compile_denoise_fill_active"] == 0.0
    assert session.last_timing["compile_denoise_masked_fill_active"] == 0.0
    assert session.last_timing["compile_denoise_static_active"] == 1.0


@pytest.mark.parametrize("text", [False, True])
def test_fixed_shape_fill_matches_eager_across_eviction_and_reset(
    monkeypatch, text: bool
) -> None:
    """Masking padding must preserve output before/after eviction and reset."""
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    shared = _net()
    window = 2 * _CHUNK
    eager_session = _session(
        shared=shared,
        text=text,
        window_frames=window,
        cache_mode="block",
    )
    fixed_session = _session(
        shared=shared,
        text=text,
        window_frames=window,
        cache_mode="block",
        compile_denoise=True,
        fixed_shape_cache_fill=True,
    )

    def assert_matches(count: int) -> None:
        expected = _steps(eager_session, count)
        actual = _steps(fixed_session, count)
        for expected_frame, actual_frame in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                actual_frame,
                expected_frame,
                rtol=2e-4,
                atol=2e-4,
            )

    assert_matches(4)
    eager_session.reset_rollout()
    fixed_session.reset_rollout()
    assert_matches(2)


def test_dynamic_fill_matches_eager_through_the_window_boundary(monkeypatch) -> None:
    """The faster fill path must preserve history before and after eviction."""
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    shared = _net()
    window = 2 * _CHUNK
    eager = _steps(_session(shared=shared, window_frames=window, cache_mode="block"), 4)
    compiled = _steps(
        _session(
            shared=shared,
            window_frames=window,
            cache_mode="block",
            compile_denoise=True,
        ),
        4,
    )
    for index, (expected, actual) in enumerate(zip(eager, compiled, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


def test_the_compiled_packer_matches_the_masked_one() -> None:
    """The structural interleave must reproduce ``_assemble_input`` exactly.

    The compiled path rebuilds the ``[action, vision]`` supertoken by
    concatenation rather than boolean assignment, and reads the velocity out by
    slicing rather than masking. Both are easy to get subtly wrong, and the
    result would be a scrambled token stream rather than an error.
    """
    shared = _net()
    window = 4 * _CHUNK
    eager = _steps(_session(shared=shared, window_frames=window, cache_mode="block"), 1)
    session = _session(
        shared=shared, window_frames=window, cache_mode="block", compile_denoise=True
    )
    # Force the static steady-state callable on immediately so its packer is
    # compared with eager execution over the same first step.
    session._compile_start_step = 0
    compiled = _steps(session, 1)
    for index, (expected, actual) in enumerate(zip(eager, compiled, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


@pytest.mark.parametrize("fixed_shape_cache_fill", [False, True])
def test_the_compiled_fill_matches_eager(fixed_shape_cache_fill: bool) -> None:
    """Both production compiled fill paths preserve the first visible chunk."""
    shared = _net()
    window = 4 * _CHUNK
    eager = _steps(_session(shared=shared, window_frames=window, cache_mode="block"), 1)
    session = _session(
        shared=shared,
        window_frames=window,
        cache_mode="block",
        compile_denoise=True,
        fixed_shape_cache_fill=fixed_shape_cache_fill,
    )
    compiled = _steps(session, 1)
    for index, (expected, actual) in enumerate(zip(eager, compiled, strict=True)):
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-4, msg=f"frame {index}"
        )


def test_the_compiled_path_denoises_at_every_sigma() -> None:
    """Each denoise step has its own sigma, so it cannot be a captured constant.

    Passing the timestep as a Python float would let a CUDA graph bake in the
    first sigma, which silently degrades output rather than failing.
    """
    shared = _net()
    session = _session(
        shared=shared,
        window_frames=4 * _CHUNK,
        cache_mode="block",
        compile_denoise=True,
    )
    session._compile_start_step = 0
    seen: list[float] = []
    inner = session._denoise_step_fn

    def recording(latent, timestep, *args):
        seen.append(float(torch.as_tensor(timestep).reshape(-1)[0]))
        return inner(latent, timestep, *args)

    session._denoise_step_fn = recording
    with torch.no_grad():
        session.step(_actions(_CHUNK))
    assert len(seen) == session.model.scheduler.config.num_inference_steps
    assert len(set(seen)) == len(seen), f"sigma did not vary across steps: {seen}"
    assert all(torch.is_tensor(torch.as_tensor(value)) for value in seen)


def test_overlap_degrades_to_synchronous_off_cuda() -> None:
    """The flag has to be safe to leave on in a CPU test or a CPU deployment."""
    session = _session(
        window_frames=_CHUNK, cache_mode="block", overlap_cache_update=True
    )
    assert session._cache_stream is None
    with torch.no_grad():
        assert len(session.step(_actions(_CHUNK))) == _CHUNK
    assert session.wait_for_pending_cache_update() == 0.0


def test_distilled_schedules_match_the_trained_two_and_four_step_models() -> None:
    two = FlowMatchSchedulerConfig(
        num_inference_steps=2,
        shift=5.0,
        denoising_timesteps=[1000, 500],
    ).setup()
    four = FlowMatchSchedulerConfig(shift=5.0).setup()
    torch.testing.assert_close(two.denoising_sigmas, torch.tensor([1.0, 5.0 / 6.0]))
    torch.testing.assert_close(
        four.denoising_sigmas,
        torch.tensor([1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0]),
    )


@pytest.mark.parametrize(
    "dispatch",
    [
        "eager",
        "compiled_fill",
        "compiled_steady",
        "graph_fill",
        "graph_steady",
        "graph_drain",
    ],
)
@pytest.mark.parametrize("steps", [2, 4])
@torch.inference_mode()
def test_bf16_flow_preserves_fp32_sde_sample(
    monkeypatch: pytest.MonkeyPatch, dispatch: str, steps: int
) -> None:
    """Promote checkpoint velocity before the scheduler multiplies by sigma."""
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    net, spec = _net()
    net.to(dtype=torch.bfloat16)
    net.config.dtype = torch.bfloat16
    session = _session(
        shared=(net, spec),
        window_frames=2 * _CHUNK,
        cache_mode="cat" if dispatch == "eager" else "block",
        compile_denoise=dispatch != "eager",
    )
    transformer = session.model.transformer
    cache = session.cache_object
    assert isinstance(transformer, HandTransformer)
    assert isinstance(cache, HandTransformerCache)
    if dispatch.endswith("_steady"):
        cache._compile_start_step = 0
    cache.prepare(_actions(_CHUNK))
    if dispatch == "graph_fill":
        # Exercise the graph wrapper's CPU warmup path, not CUDA capture.
        cache._denoise_fill_graph = CUDAGraphWrapper(
            cache._denoise_fill_fn, warmup_iters=steps
        )
    elif dispatch in {"graph_steady", "graph_drain"}:
        cache._denoise_graph = CUDAGraphWrapper(
            cache._denoise_step_fn, warmup_iters=steps
        )
        if dispatch == "graph_drain":
            cache._compiled_fill_active = False

    raw_flows: list[torch.Tensor] = []

    def record_flow(
        _module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        assert output.dtype == torch.bfloat16
        assert cache._is_action is not None
        raw_flows.append(
            cache._to_logical(output[0][~cache._is_action], cache._frames)
            .unsqueeze(0)
            .clone()
        )

    def predict_flow(noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        flow = transformer.predict_flow(noisy, timestep, cache)
        assert flow.dtype == noisy.dtype == torch.float32
        torch.testing.assert_close(flow, raw_flows[-1].float(), rtol=0, atol=0)
        return flow

    scheduler = FlowMatchSchedulerConfig(
        num_inference_steps=steps,
        shift=5.0,
        denoising_timesteps=[1000, 500] if steps == 2 else [1000, 750, 500, 250],
    ).setup()
    initial = torch.randn(
        1,
        cache.channels,
        cache._frames,
        cache.h_lat,
        cache.w_lat,
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float32,
    )
    hook = net.llm2vae.register_forward_hook(record_flow)
    try:
        actual = scheduler.sample(
            initial_noise=initial,
            predict_flow=predict_flow,
            rng=torch.Generator().manual_seed(123),
        )
    finally:
        hook.remove()

    assert len(raw_flows) == steps
    rng = torch.Generator().manual_seed(123)
    expected = initial
    for index, (sigma, flow) in enumerate(
        zip(scheduler.denoising_sigmas, raw_flows, strict=True)
    ):
        if index:
            noise = torch.empty_like(initial).normal_(generator=rng)
            expected = (1.0 - sigma) * expected + sigma * noise
        expected = expected - sigma * flow.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_logical_latent_geometry_must_fit_the_grid() -> None:
    net, spec = _net()
    init = torch.zeros(_GRID[0] * _GRID[1], spec.vfm.patch_latent_dim)
    with pytest.raises(ValueError, match="logical latent geometry"):
        HandTransformerCache(
            net,
            scene=SimpleNamespace(name="test"),
            text_ids=None,
            init_latent=init,
            grid_hw=_GRID,
            chunk_size=_CHUNK,
            tcf=_TCF,
            logical_latent_hw=(99, 99),
        )
