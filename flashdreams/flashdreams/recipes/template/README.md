# `template` recipe

Minimal end-to-end recipe exercising every contract in
`flashdreams.infra`. Use as a reference when scaffolding a new recipe.

## What's exercised

- **Offline rollout** — `build_cfg_offline`, one AR step over the full
  temporal window.
- **Autoregressive streaming** — `build_cfg_autoregressive`, multiple
  AR steps over a sliding `BlockKVCache`.
- **Classifier-free guidance** — opt-in by patching
  `guidance_scale > 1.0` via `derive_config`. The uncond branch gets an
  independent `network_cache_uncond`; `negative_context` is encoded
  only when CFG is on.
- **Context parallelism** along `L` — `cp_size` auto-detected from
  `torch.distributed.get_world_size()`, so `torchrun --nproc_per_node=N`
  is the single source of truth. Attention uses
  `flashdreams.core.attention.RingAttention` (fuses the KV gather with
  SDPA via an LSE merge).
- **Per-AR-step control input** — `TemplateControlEncoder` (a
  `StreamingEncoder`). Setting `encoder=None` exercises the no-control
  branch (`test_template_no_control`).
- **One-shot context encoder** — `TemplateTransformerConfig.context_encoder`
  (an `Encoder`; defaults to `NullEncoderConfig`, identity). Plug in a
  text or CLIP image encoder here.
- **Output decoding** — `TemplateDecoder` (a `StreamingDecoder` that
  ignores its empty cache; swap for a `StreamingVideoDecoder` subclass
  when the recipe needs spatial / temporal compression contracts).
- **Config derivation** — `build_cfg_autoregressive` is a
  `derive_config` patch on top of `build_cfg_offline`.
- **`torch.compile` + `CUDAGraphWrapper`** — off by default; flip via
  `with_compile_and_cuda_graph` (or `derive_config`). Covered by
  `test_template_compile_and_cudagraph_equivalence`.
- **Checkpoint loading** — `TemplateTransformerConfig.checkpoint_path`.
  `None` keeps the random init.

## Files

| Path | Purpose |
|---|---|
| `transformer/network.py` | 1-block DiT + `BlockKVCache` plumbing. |
| `transformer/__init__.py` | `Transformer` subclass, AR cache, config. |
| `encoder.py` | Control encoder (control channels → latent channels). |
| `decoder.py` | Decoder (latent channels → output channels). |
| `config.py` | Pipeline builders + `with_compile_and_cuda_graph`. |

## Entry point

```bash
uv run --extra dev pytest flashdreams/tests/test_template.py -v
# or as a script
uv run --extra dev python flashdreams/tests/test_template.py
# multi-GPU CP equivalence (after the single-GPU run has written the reference)
uv run --extra dev torchrun --nproc_per_node=2 -m pytest \
    flashdreams/tests/test_template.py::test_template_cp_equivalence -v
```

## Shape conventions

- Pre-patchify: `[B, C, T, H, W]` for both noisy latent and control.
- Post-patchify: `[B, L=T*H*W, C]`, CP-split to `[B, L/cp_size, C]`.
- Decoded output: `[B, C_out, T, H, W]`.
