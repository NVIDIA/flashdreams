# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the independently authored Waypoint adapter."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from waypoint import (
    WAYPOINT_1_5,
    WaypointControl,
    load_controls_from_file,
    make_control_context,
)
from waypoint.config import PIPELINE_WAYPOINT_1_5
from waypoint.impl.checkpoint import (
    expected_waypoint_1_5_checkpoint_keys,
    expected_waypoint_1_5_checkpoint_shapes,
    load_waypoint_state_dict,
    validate_waypoint_1_5_checkpoint_keys,
    validate_waypoint_1_5_checkpoint_shapes,
)
from waypoint.impl.decoder import WaypointTAEHVDecoder
from waypoint.impl.encoder import WaypointControlEncoderConfig
from waypoint.impl.pipeline import (
    WaypointInferencePipeline,
    WaypointInferencePipelineConfig,
)
from waypoint.impl.transformer import (
    WaypointAttentionPolicy,
    WaypointDiTConfig,
    WaypointKVCache,
    WaypointOrthoRoPEAngles,
    WaypointTransformerConfig,
    adaptive_gate,
    adaptive_rms_norm,
    apply_waypoint_ortho_rope,
)
from waypoint.impl.transformer.impl import WaypointTransformerCache
from waypoint.impl.transformer.network import (
    _ConditionHead,
    _ControlFusion,
    _WaypointAttention,
    _WaypointBlock,
    sinusoidal_noise_embedding,
)

pytestmark = pytest.mark.ci_cpu


def test_waypoint_1_5_static_contract() -> None:
    """The published Waypoint configuration has stable model-shape invariants."""
    assert isinstance(PIPELINE_WAYPOINT_1_5, WaypointInferencePipelineConfig)
    assert WAYPOINT_1_5.latent_shape() == (1, 1, 32, 32, 64)
    assert WAYPOINT_1_5.tokens_per_latent_frame == 512
    assert WAYPOINT_1_5.patch_grid_height == 16
    assert WAYPOINT_1_5.patch_grid_width == 32
    assert WAYPOINT_1_5.frames_per_action == 4
    assert WAYPOINT_1_5.num_denoising_steps == 4
    assert WAYPOINT_1_5.frame_timestamp_stride == 1
    assert WAYPOINT_1_5.head_dim == 64
    assert WAYPOINT_1_5.global_attention_layers == (3, 7, 11, 15, 19, 23)
    assert WAYPOINT_1_5.global_pinned_dilation == 8
    assert WAYPOINT_1_5.value_residual
    assert WAYPOINT_1_5.noise_conditioning == "wan"
    assert WAYPOINT_1_5.noise_embedding_dim == 512
    assert WAYPOINT_1_5.rope_theta == 10_000.0
    assert WAYPOINT_1_5.rope_nyquist_fraction == 0.8
    assert not WAYPOINT_1_5.text_conditioning


def test_control_context_preserves_waypoint_action_semantics() -> None:
    """Buttons, motion, scroll, and latent-frame time use model-ready shapes."""
    context = make_control_context(
        WaypointControl(buttons=frozenset({65, 87}), mouse_dx=0.125, mouse_dy=-0.25),
        frame_index=7,
        dtype=torch.float32,
    )
    assert context["button"].shape == (1, 1, 256)
    assert context["button"][0, 0, 65].item() == 1.0
    assert context["button"][0, 0, 87].item() == 1.0
    assert context["mouse"].tolist() == [[[0.125, -0.25]]]
    assert context["scroll"].tolist() == [[[0.0]]]
    assert context["frame_idx"].tolist() == [[7]]
    assert context["frame_timestamp"].tolist() == [[7]]


def test_control_timeline_file_preserves_per_action_inputs(tmp_path) -> None:
    """JSON control files retain neutral defaults and explicit action values."""
    path = tmp_path / "controls.json"
    path.write_text(
        """{
  "schema_version": 1,
  "actions": [
    {"buttons": [32], "mouse_dx": 0.1, "scroll_wheel": -1},
    {}
  ]
}
""",
        encoding="utf-8",
    )
    controls = load_controls_from_file(path)
    assert controls == (
        WaypointControl(buttons=frozenset({32}), mouse_dx=0.1, scroll_wheel=-1),
        WaypointControl(),
    )


@pytest.mark.parametrize("buttons", [frozenset({-1}), frozenset({256})])
def test_control_context_rejects_invalid_button_ids(buttons: frozenset[int]) -> None:
    """The native adapter must never index outside the fixed button vocabulary."""
    with pytest.raises(ValueError, match="button IDs"):
        make_control_context(WaypointControl(buttons=buttons), frame_index=0)


def test_waypoint_raw_checkpoint_schema_accounts_for_every_tensor() -> None:
    """The checkpoint schema captures the published raw 393-tensor layout."""
    keys = expected_waypoint_1_5_checkpoint_keys()
    assert len(keys) == 393
    validate_waypoint_1_5_checkpoint_keys(keys)


def test_waypoint_raw_checkpoint_schema_rejects_unknown_tensor() -> None:
    """Loader work must not silently accept a changed checkpoint format."""
    keys = set(expected_waypoint_1_5_checkpoint_keys())
    keys.add("transformer.blocks.0.unknown.weight")
    with pytest.raises(ValueError, match="extra"):
        validate_waypoint_1_5_checkpoint_keys(keys)


def test_raw_checkpoint_shapes_match_the_published_schema() -> None:
    """The checkpoint contract records every raw tensor shape without a fake network."""
    shapes = expected_waypoint_1_5_checkpoint_shapes()
    assert set(shapes) == expected_waypoint_1_5_checkpoint_keys()
    assert shapes["patchify.weight"] == (2048, 32, 2, 2)
    assert shapes["unpatchify.weight"] == (2048, 32, 2, 2)
    assert shapes["transformer.blocks.0.attn.k_proj.weight"] == (1024, 2048)
    validate_waypoint_1_5_checkpoint_shapes(shapes)


def test_checkpoint_loader_requires_the_exact_native_namespace() -> None:
    """Validated raw tensors load without a hidden key remap."""
    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=1,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        mlp_ratio=2,
    )
    source = WaypointDiTConfig(spec=tiny_spec).setup()
    target = WaypointDiTConfig(spec=tiny_spec).setup()
    state_dict = source.state_dict()
    load_waypoint_state_dict(target, state_dict, spec=tiny_spec)
    assert torch.equal(target.patchify.weight, source.patchify.weight)


def test_raw_checkpoint_shape_validation_rejects_a_mismatch() -> None:
    """Loader work must not accept a changed raw tensor shape."""
    shapes = expected_waypoint_1_5_checkpoint_shapes()
    shapes["patchify.weight"] = (1,)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_waypoint_1_5_checkpoint_shapes(shapes)


def test_native_dit_topology_matches_the_raw_checkpoint_schema() -> None:
    """The native DiT module graph can load every published raw tensor name."""
    with torch.device("meta"):
        network = WaypointDiTConfig().setup()
    state_shapes = {
        key: tuple(value.shape) for key, value in network.state_dict().items()
    }
    assert state_shapes == expected_waypoint_1_5_checkpoint_shapes()
    query, key, value = network.transformer.blocks[0].attn.project_qkv(
        torch.empty(2, 512, WAYPOINT_1_5.d_model, device="meta")
    )
    assert query.shape == (2, 512, 32, 64)
    assert key.shape == value.shape == (2, 512, 16, 64)


def test_orthogonal_rope_angles_match_waypoint_geometry() -> None:
    """RoPE reserves x, y, and time frequency bands in that exact order."""
    rope = WaypointOrthoRoPEAngles()
    cosine, sine = rope(
        frame_index=torch.tensor([0, 1]),
        row_index=torch.tensor([0, 1]),
        column_index=torch.tensor([0, 1]),
    )
    assert cosine.shape == (2, 1, 32)
    assert sine.shape == (2, 1, 32)
    # At x=0, the first frequency is pi / 16 at the centered patch x=-15.5.
    assert cosine[0, 0, 0].item() == pytest.approx(-0.9951847, abs=1e-6)
    assert sine[0, 0, 0].item() == pytest.approx(-0.0980171, abs=1e-6)
    # The temporal band follows the two eight-value spatial bands.
    assert cosine[1, 0, 16].item() == pytest.approx(torch.cos(torch.tensor(1.0)).item())
    assert sine[1, 0, 16].item() == pytest.approx(torch.sin(torch.tensor(1.0)).item())


def test_wan_noise_features_use_waypoints_scaled_sine_cosine_order() -> None:
    """Noise modulation uses the checkpoint's 1000x, sqrt-two Fourier basis."""
    features = sinusoidal_noise_embedding(512, torch.tensor([1.0]))
    assert features.shape == (1, 512)
    assert features[0, 0].item() == pytest.approx(1.1693842, abs=1e-6)
    assert features[0, 256].item() == pytest.approx(0.7953241, abs=1e-6)


def test_orthogonal_rope_rotates_two_half_heads() -> None:
    """RoPE rotates each packed half-head pair without mixing attention heads."""
    tokens = torch.zeros(1, 1, 1, 64)
    tokens[..., 0] = 1.0
    cosine = torch.full((1, 1, 32), 0.6)
    sine = torch.full((1, 1, 32), 0.8)
    output = apply_waypoint_ortho_rope(tokens, cosine, sine)
    assert output[..., 0].item() == pytest.approx(0.6)
    assert output[..., 32].item() == pytest.approx(0.8)


def test_adaptive_rms_norm_conditions_each_latent_frame() -> None:
    """AdaRMSNorm broadcasts a separate scale and bias over each frame's tokens."""
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [2.0, 4.0], [6.0, 8.0]]])
    scale = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    bias = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    output = adaptive_rms_norm(tokens, scale, bias)
    rms = torch.rsqrt(torch.mean(tokens.square(), dim=-1, keepdim=True))
    expected = tokens * rms
    expected[:, :2] = expected[:, :2] * torch.tensor([2.0, 1.0]) + torch.tensor(
        [0.1, 0.2]
    )
    expected[:, 2:] = expected[:, 2:] * torch.tensor([1.0, 2.0]) + torch.tensor(
        [0.3, 0.4]
    )
    assert torch.allclose(output, expected)


def test_adaptive_gate_scales_each_latent_frame() -> None:
    """AdaGate broadcasts each frame's learned residual multiplier over its tokens."""
    tokens = torch.ones(1, 4, 2)
    gate = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
    output = adaptive_gate(tokens, gate)
    assert torch.equal(
        output,
        torch.tensor([[[2.0, 3.0], [2.0, 3.0], [5.0, 7.0], [5.0, 7.0]]]),
    )


def test_value_residual_blends_with_the_first_block_value_stream() -> None:
    """Later blocks linearly blend their V tensor with the retained first V tensor."""
    attention = _WaypointAttention(
        replace(WAYPOINT_1_5, d_model=8, n_heads=1, n_kv_heads=1)
    )
    current = torch.tensor([[[[1.0, 2.0]]]])
    initial = torch.tensor([[[[10.0, 20.0]]]])
    attention.v_lamb.data.fill_(2.0)
    mixed, retained = attention.blend_value_residual(current, initial)
    assert torch.equal(mixed, torch.tensor([[[[19.0, 38.0]]]]))
    assert retained is initial


def test_attention_uses_grouped_query_sparse_kv_view() -> None:
    """Attention accepts cached KV heads without expanding them to query heads."""
    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    attention = _WaypointAttention(tiny_spec)
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        projection.weight.data.copy_(torch.eye(16))
    attention.v_lamb.data.zero_()
    cache = WaypointKVCache(policy=WaypointAttentionPolicy(spec=tiny_spec))
    tokens = torch.tensor([[[1.0] * 16, [2.0] * 16]])
    cosine = torch.ones(2, 1, 8)
    sine = torch.zeros(2, 1, 8)
    output, retained = attention(
        tokens,
        cosine=cosine,
        sine=sine,
        layer_index=0,
        frame_index=0,
        kv_cache=cache,
        initial_value=None,
    )
    assert output.shape == tokens.shape
    assert retained.shape == (1, 2, 1, 16)


def test_block_composes_adaptive_attention_control_and_mlp_paths() -> None:
    """The block preserves frame-aware conditioning through both residual paths."""
    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        mlp_ratio=2,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    block = _WaypointBlock(tiny_spec, has_control_fusion=True)
    cache = WaypointKVCache(policy=WaypointAttentionPolicy(spec=tiny_spec))
    tokens = torch.randn(1, 2, 16)
    conditioning = torch.randn(1, 1, 16)
    control = torch.randn(1, 2, 16)
    output, initial_value = block(
        tokens,
        conditioning=conditioning,
        control=control,
        cosine=torch.ones(2, 1, 8),
        sine=torch.zeros(2, 1, 8),
        layer_index=0,
        frame_index=0,
        kv_cache=cache,
        initial_value=None,
    )
    assert output.shape == tokens.shape
    assert initial_value.shape == (1, 2, 1, 16)


def test_dit_runs_one_complete_latent_action() -> None:
    """The native DiT keeps controller, RoPE, cache, and output layouts aligned."""
    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        mlp_ratio=2,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    network = WaypointDiTConfig(spec=tiny_spec).setup()
    for parameter in network.parameters():
        parameter.data.zero_()
    cache = WaypointKVCache(policy=WaypointAttentionPolicy(spec=tiny_spec))
    latent = torch.randn(1, 1, 32, 32, 64)
    output = network(
        latent,
        sigma=torch.ones(1),
        frame_index=0,
        kv_cache=cache,
        button=torch.zeros(1, 1, 256),
        mouse=torch.zeros(1, 1, 2),
        scroll=torch.zeros(1, 1, 1),
    )
    assert output.shape == latent.shape
    assert torch.equal(output, torch.zeros_like(latent))


def test_flashdreams_transformer_adapter_owns_action_layout_and_control() -> None:
    """The framework adapter passes one public action into the native DiT."""
    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        mlp_ratio=2,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    transformer = WaypointTransformerConfig(
        network=WaypointDiTConfig(spec=tiny_spec), dtype=torch.float32
    ).setup()
    for parameter in transformer.parameters():
        parameter.data.zero_()
    cache = transformer.initialize_autoregressive_cache(batch_size=1)
    cache.start(0)
    noisy = torch.randn(1, 1, 32, 32, 64)
    output = transformer.predict_flow(
        noisy,
        torch.tensor(1.0),
        cache,
        WaypointControl(buttons=frozenset({87})),
    )
    assert output.shape == noisy.shape
    assert torch.equal(output, torch.zeros_like(noisy))
    external = transformer.unpatchify_and_maybe_gather_cp(output)
    assert external.shape == (1, 32, 1, 32, 64)
    assert torch.equal(transformer.patchify_and_maybe_split_cp(external), output)


def test_pipeline_runs_fixed_euler_steps_for_one_controlled_action() -> None:
    """The generic pipeline can drive Waypoint without a custom serving loop."""
    from flashdreams.infra.diffusion.model import DiffusionModelConfig
    from flashdreams.infra.diffusion.scheduler import (
        FlowMatchEulerDiscreteSchedulerConfig,
    )
    from flashdreams.infra.pipeline import StreamInferencePipelineConfig

    tiny_spec = replace(
        WAYPOINT_1_5,
        n_layers=1,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        mlp_ratio=2,
    )
    pipeline = StreamInferencePipelineConfig(
        name="waypoint-test",
        diffusion_model=DiffusionModelConfig(
            transformer=WaypointTransformerConfig(
                network=WaypointDiTConfig(spec=tiny_spec), dtype=torch.float32
            ),
            scheduler=FlowMatchEulerDiscreteSchedulerConfig(
                num_inference_steps=4,
                num_train_timesteps=1,
                fixed_timesteps=tiny_spec.scheduler_sigmas,
            ),
        ),
        encoder=WaypointControlEncoderConfig(),
    ).setup()
    for parameter in pipeline.parameters():
        parameter.data.zero_()
    cache = pipeline.initialize_cache(transformer_context={"batch_size": 1})
    output = pipeline.generate(0, cache, WaypointControl(buttons=frozenset({87})))
    pipeline.finalize(0, cache)
    assert output.shape == (1, 32, 1, 32, 64)


def test_seed_initialization_advances_decoder_history_once() -> None:
    """The seed latent establishes one matching transformer and decoder action."""
    pipeline = WaypointInferencePipeline.__new__(WaypointInferencePipeline)
    torch.nn.Module.__init__(pipeline)

    seed_latent = torch.zeros(1, 1, 32, 32, 64)
    decoder_cache = object()
    decoder = WaypointTAEHVDecoder.__new__(WaypointTAEHVDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.initialize_autoregressive_cache = Mock(return_value=decoder_cache)
    decoder.forward = Mock()

    transformer_cache = WaypointTransformerCache(batch_size=1)
    transformer = SimpleNamespace(
        initialize_autoregressive_cache=Mock(return_value=transformer_cache),
        predict_flow=Mock(),
    )
    pipeline.seed_encoder = SimpleNamespace(
        taehv=SimpleNamespace(encode=Mock(return_value=seed_latent))
    )
    pipeline.encoder = None
    pipeline.decoder = decoder
    pipeline.diffusion_model = SimpleNamespace(
        device=torch.device("cpu"), transformer=transformer
    )

    cache = pipeline.initialize_cache(
        seed_pixels=torch.zeros(1, 4, 3, 512, 1024),
    )

    transformer.predict_flow.assert_called_once()
    decoder.forward.assert_called_once()
    assert decoder.forward.call_args.args[0].shape == (1, 32, 1, 32, 64)
    assert decoder.forward.call_args.kwargs == {
        "autoregressive_index": 0,
        "cache": decoder_cache,
    }
    assert cache.autoregressive_index == 0


def test_attention_policy_selects_dense_local_and_pinned_global_history() -> None:
    """Waypoint uses a short dense history or a dilated long-range history."""
    policy = WaypointAttentionPolicy()
    assert not policy.is_global_layer(2)
    assert policy.is_global_layer(3)
    assert policy.visible_frame_indices(layer_index=2, frame_index=17) == tuple(
        range(2, 18)
    )
    assert policy.visible_frame_indices(layer_index=3, frame_index=130) == (
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        130,
    )


def test_kv_cache_replaces_provisional_frame_and_evicts_hidden_history() -> None:
    """A cache view contains only the frames the checkpoint can attend to."""
    small_spec = replace(
        WAYPOINT_1_5,
        n_layers=4,
        local_window=2,
        global_window=6,
        global_pinned_dilation=2,
    )
    cache = WaypointKVCache(policy=WaypointAttentionPolicy(spec=small_spec))

    for frame_index in range(7):
        value = torch.full((1, 1, 2, 2), float(frame_index))
        global_view = cache.update(
            layer_index=3,
            frame_index=frame_index,
            key=value,
            value=-value,
        )
    assert global_view.frame_indices == (2, 4, 6)
    assert global_view.key[0, 0, ::2, 0].tolist() == [2.0, 4.0, 6.0]

    replacement = torch.full((1, 1, 2, 2), 99.0)
    global_view = cache.update(
        layer_index=3,
        frame_index=6,
        key=replacement,
        value=-replacement,
    )
    assert global_view.key[0, 0, ::2, 0].tolist() == [2.0, 4.0, 99.0]

    for frame_index in range(4):
        value = torch.full((1, 1, 2, 2), float(frame_index))
        local_view = cache.update(
            layer_index=0,
            frame_index=frame_index,
            key=value,
            value=value,
        )
    assert local_view.frame_indices == (2, 3)
    assert local_view.key[0, 0, ::2, 0].tolist() == [2.0, 3.0]


def test_condition_and_control_fusion_primitives_use_silu_paths() -> None:
    """Checkpoint projection names retain the observed conditioning semantics."""
    condition = _ConditionHead(2)
    condition.bias_in.data.copy_(torch.tensor([0.5, -0.5]))
    for projection in condition.cond_proj:
        assert isinstance(projection, torch.nn.Linear)
        projection.weight.data.copy_(torch.eye(2))
    values = condition(torch.tensor([[-1.0, 2.0]]))
    expected = torch.nn.functional.silu(torch.tensor([[-0.5, 1.5]]))
    assert all(torch.allclose(value, expected) for value in values)

    fusion = _ControlFusion(2)
    fusion.fc1_x.weight.data.copy_(torch.eye(2))
    fusion.fc1_c.weight.data.copy_(torch.eye(2))
    fusion.fc2.weight.data.copy_(torch.eye(2))
    output = fusion(torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[3.0, 4.0]]]))
    assert torch.allclose(
        output, torch.nn.functional.silu(torch.tensor([[[4.0, 6.0]]]))
    )
