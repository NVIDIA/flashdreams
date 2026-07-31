# Stream Lingbot Sharded Safetensors Loading

## Summary

This PR works around Lingbot single-GPU system RAM pressure by loading Hugging Face sharded safetensors checkpoints one shard at a time. The Lingbot DiT now builds on `meta` for streamable sharded checkpoints and materializes weights directly on the runner CUDA device, avoiding the previous full CPU module plus full CPU state-dict peak.

## Changes

- Add a streamed sharded safetensors checkpoint loader that resolves Hugging Face or local `*.safetensors.index.json` files, downloads shard files, loads one shard at a time, and validates missing, unexpected, and remaining meta keys.
- Avoid extra host copies for local safetensors by using `safetensors.torch.load_file`.
- Use thread-based Hugging Face shard downloads instead of process workers, avoiding extra Python/Torch worker RSS during checkpoint download.
- Add `checkpoint_map_location` to Wan transformer configs and use the streamed loader for sharded safetensors without `state_dict_transform`.
- Resolve Lingbot transformer `checkpoint_map_location="auto"` to the runner CUDA target, including torchrun/local-rank cases.
- Load UMT5 and CLIP encoders with `low_cpu_mem_usage=True` and the configured dtype.
- Add focused CPU tests for streamed sharded loading, Wan meta-load wiring, Lingbot CUDA map-location resolution, and low-RAM encoder load kwargs.

## Benchmark Data

Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB total VRAM, 62 GiB host RAM, 8 GiB swap.

Runner: `RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3`, `example_data=True`, `total_blocks=1`, 832x464, 9 output frames.

| Case | Result | Host RSS | GPU Memory | Notes |
| --- | --- | ---: | ---: | --- |
| Previous full CPU load path | Failed, exit 137 | ~54,747,488 KB | No meaningful GPU allocation | Killed during CPU-side model load. |
| Streamed direct-to-CUDA, default compile/CUDA graph | Completed | 34,466,668 KB peak | 69.236 GiB AR peak allocated; ~73.8 GiB observed process peak | Fit on one GPU, but produced NaNs/all-black frames on this Blackwell stack. |
| Streamed direct-to-CUDA, `compile_network=False`, `use_cuda_graph=False` | Completed | 35,224,172 KB peak | 69.236 GiB AR peak allocated; 73,782 MiB observed process peak | Produced a valid H.264 MP4 with 9 readable nonzero frames. |

Eager-path per-step timings:

```text
encode 2797.579 ms
diffuse 7606.392 ms
decode 20833.270 ms
finalize 474.524 ms
total 31711.765 ms
GPU mem alloc 56.958 GiB
GPU mem reserved 62.895 GiB
GPU mem peak 69.236 GiB
```

## Notes

- The system RAM workaround is the streamed direct-to-CUDA checkpoint path. The compiled/CUDA-graph Lingbot path still needs separate numerical follow-up on Blackwell because it fits but generated NaNs in this local probe.
- The generated benchmark artifacts were local probe outputs under `outputs/lingbot_single_rtx6000_stream_sharded_*` and are not committed.

## Tests

Local verification:

```bash
uv run --package flashdreams-lingbot --with pytest --with pytest-manual-marker pytest flashdreams/tests/test_checkpoint_load.py flashdreams/tests/test_wan_context_parallel.py flashdreams/tests/test_transformers_encoder_loading.py integrations/lingbot/tests/test_smoke.py -q
```

Result:

```text
27 passed in 6.77s
```
