# Reproduce QVG In FlashDreams

This is the current QVG reproduction handbook for the `kv-compress` branch.
Use the cleanup run below for review. Older pre-cleanup runs are historical
references only.

## Current Cleanup Run

Run date: 2026-05-09

```text
report: outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/qvgbench_cleanup_4prompts_seed42_genonly_20260509_8metric_report.json
log:    outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/run_benchmark.log
```

The run regenerated official BF16, official INT2, FlashDreams BF16, and
FlashDreams INT2 for prompt indices `0,1,2,3`, seed `42`, 16 AR blocks, and
189 decoded frames after the cleanup pass. It also wrote four labeled 2x2 grid
videos:

```text
outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/grids/qvgcleanup_genonly_grid_prompt0.mp4
outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/grids/qvgcleanup_genonly_grid_prompt1.mp4
outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/grids/qvgcleanup_genonly_grid_prompt2.mp4
outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/grids/qvgcleanup_genonly_grid_prompt3.mp4
```

Grid layout is top row `Official BF16`, `Official INT2`; bottom row
`FlashDreams BF16`, `FlashDreams INT2`. Each panel has a top-left label.

The table below reports raw values only. The benchmark JSON still carries
diagnostic tolerance fields, but this handbook does not use yes/no pass columns.

Metric semantics:

- PSNR, SSIM, and LPIPS are full-video metrics over decoded frames `[0, 189)`.
- VBench metrics are over the whole generated video.
- VBench rows are formatted as `BF16 score (INT2 delta)`.
- Compression ratio is INT2 persistent KV footprint versus BF16 KV footprint.
- Generation FPS is diagnostic, formatted as `BF16 fps (INT2 delta)`.
- End-to-end wall seconds/FPS stay in JSON only as diagnostics because they
  include process startup, checkpoint loading, video IO, and benchmark
  bookkeeping.

## Average Metrics

| Metric | Official QVG | FlashDreams QVG |
| --- | ---: | ---: |
| Compression ratio | 6.6000 | 6.6064 |
| PSNR full video | 18.3670 | 21.9024 |
| SSIM full video | 0.7401 | 0.8315 |
| LPIPS full video | 0.1434 | 0.0755 |
| VBench Background Consistency | 0.9294 (+0.0032) | 0.9460 (+0.0011) |
| VBench Image Quality | 0.7186 (+0.0040) | 0.7261 (-0.0049) |
| VBench Subject Consistency | 0.9274 (+0.0086) | 0.9353 (-0.0023) |
| VBench Aesthetic Quality | 0.6416 (+0.0011) | 0.6477 (-0.0004) |
| Generation FPS | 2.8268 (-0.3844) | 5.1844 (-1.7730) |

## Per-Prompt Metrics

Prompt 0:

```text
realistic filming style, a street food vendor flips thin pancakes on a sizzling griddle at night in a crowded market. Steam rises into colorful lantern light while customers wait in line, cyclists pass behind the stall, and the camera slowly tracks from left to right at counter height.
```

| Metric | Official QVG | FlashDreams QVG |
| --- | ---: | ---: |
| Compression ratio | 6.6000 | 6.6064 |
| PSNR full video | 17.5408 | 19.8773 |
| SSIM full video | 0.6853 | 0.7510 |
| LPIPS full video | 0.1643 | 0.0985 |
| VBench Background Consistency | 0.9447 (-0.0325) | 0.9419 (+0.0168) |
| VBench Image Quality | 0.7128 (+0.0502) | 0.7076 (-0.0162) |
| VBench Subject Consistency | 0.9181 (-0.0298) | 0.9537 (-0.0315) |
| VBench Aesthetic Quality | 0.6419 (-0.0017) | 0.6242 (+0.0218) |
| Generation FPS | 2.5022 (-0.2669) | 4.7957 (-1.6009) |

Prompt 1:

```text
cinematic wildlife footage of a red fox walking through fresh snow in a quiet pine forest at sunrise. The fox pauses to listen, turns its head toward the camera, then trots across the frame while powder falls from branches in warm golden light.
```

| Metric | Official QVG | FlashDreams QVG |
| --- | ---: | ---: |
| Compression ratio | 6.6000 | 6.6064 |
| PSNR full video | 17.6731 | 23.8249 |
| SSIM full video | 0.7173 | 0.8787 |
| LPIPS full video | 0.1589 | 0.0529 |
| VBench Background Consistency | 0.9131 (-0.0002) | 0.9601 (-0.0334) |
| VBench Image Quality | 0.7730 (-0.1174) | 0.7340 (+0.0201) |
| VBench Subject Consistency | 0.8833 (+0.0651) | 0.9707 (-0.0782) |
| VBench Aesthetic Quality | 0.6297 (-0.0346) | 0.6743 (-0.0252) |
| Generation FPS | 2.8248 (-0.3649) | 5.0158 (-1.5293) |

Prompt 2:

```text
a macro tabletop video of a mechanical pocket watch being assembled by careful hands. Tiny brass gears, screws, and springs are placed into the open case, the balance wheel begins to oscillate, and the camera glides in a smooth close-up with shallow depth of field.
```

| Metric | Official QVG | FlashDreams QVG |
| --- | ---: | ---: |
| Compression ratio | 6.6000 | 6.6064 |
| PSNR full video | 19.0401 | 24.0895 |
| SSIM full video | 0.8666 | 0.9337 |
| LPIPS full video | 0.0905 | 0.0247 |
| VBench Background Consistency | 0.9453 (+0.0166) | 0.9570 (+0.0029) |
| VBench Image Quality | 0.7326 (+0.0043) | 0.6981 (+0.0379) |
| VBench Subject Consistency | 0.9628 (-0.0165) | 0.9239 (+0.0463) |
| VBench Aesthetic Quality | 0.6797 (-0.0203) | 0.6466 (+0.0234) |
| Generation FPS | 2.7790 (-0.4874) | 5.4281 (-1.9060) |

Prompt 3:

```text
an aerial drone shot over a rocky ocean coastline during late afternoon. Waves crash against cliffs, seabirds circle above the water, and the camera flies forward along the shore as sunlight reflects off sea spray and tide pools.
```

| Metric | Official QVG | FlashDreams QVG |
| --- | ---: | ---: |
| Compression ratio | 6.6000 | 6.6064 |
| PSNR full video | 19.2142 | 19.8179 |
| SSIM full video | 0.6911 | 0.7624 |
| LPIPS full video | 0.1600 | 0.1260 |
| VBench Background Consistency | 0.9146 (+0.0289) | 0.9249 (+0.0182) |
| VBench Image Quality | 0.6559 (+0.0786) | 0.7646 (-0.0616) |
| VBench Subject Consistency | 0.9455 (+0.0158) | 0.8927 (+0.0543) |
| VBench Aesthetic Quality | 0.6151 (+0.0610) | 0.6456 (-0.0216) |
| Generation FPS | 3.2011 (-0.4185) | 5.4979 (-2.0556) |

## How To Rerun

Start or reuse a persistent 1-GPU interactive shell. Keep the shell open after
the job so follow-up debugging can reuse it.

```bash
bash dev/slurm_interactive_ord.sh 1 --time 4:00:00
```

Inside the container:

```bash
cd /workspace/flashdreams

VBENCH_PY=/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/.cache/qvg_vbench_env/bin/python \
uv run --package flashdreams --extra examples python \
  flashdreams/examples/qvg_benchmark.py \
  --fd_dir /workspace/flashdreams \
  --qvg_dir /lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/Quant-VideoGen \
  --official_py /lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/Self-Forcing/.venv/bin/python \
  --official_config_path /workspace/flashdreams/flashdreams/examples/qvg_benchmark_assets/configs/official_self_forcing_dmd_shift8.yaml \
  --prompts /workspace/flashdreams/flashdreams/examples/qvg_benchmark_assets/prompts/qvg_prompt_matrix_extra.txt \
  --prompt_indices 0,1,2,3 \
  --seed 42 \
  --total_blocks 16 \
  --frame_count 189 \
  --num_output_frames 48 \
  --local_attn_size 180 \
  --output_dir /workspace/flashdreams/outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509 \
  --name cleanup_4prompts_seed42_genonly_20260509 \
  --tag_prefix qvgcleanup_genonly \
  --grid_stem qvgcleanup_genonly_grid
```

The runner:

1. Runs or reuses official BF16/INT2 and FlashDreams BF16/INT2 videos.
2. Crops official videos to the matched decoded length.
3. Computes full-video PSNR, SSIM, and LPIPS.
4. Runs the four VBench dimensions through `VBENCH_PY`.
5. Writes per-prompt rows, an average row, and diagnostics to the report JSON.
6. Writes one labeled 2x2 grid video per prompt.

## Implementation Notes

- `flashdreams/examples/qvg_benchmark.py` owns the current reproduction
  pipeline.
- Runtime benchmark assets live under
  `flashdreams/examples/qvg_benchmark_assets/`.
- Official QVG is launched through the local clone at
  `/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/Quant-VideoGen`.
- Official QVG uses chunkwise initial-noise compatibility by default through
  the runner's `--chunkwise-official-noise` behavior.
- FlashDreams QVG uses `kernel_impl="official_triton"` for this benchmark,
  which calls the vendored official QVG Triton PRQ/k-means kernels.
- Triton kernels are lazily JIT-compiled; no explicit precompile pass is used
  in this benchmark.
- The current FlashDreams INT2 port meets the memory goal:
  persistent KV compression ratio is `6.6064x`.
- Previous pre-cleanup review report, superseded by this run:
  `outputs/qvgbench_repro_review_4prompts_seed42_genonly_20260509/qvgbench_repro_review_4prompts_seed42_genonly_20260509_8metric_report.json`.

## Verification Commands

```bash
python -m json.tool \
  outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/qvgbench_cleanup_4prompts_seed42_genonly_20260509_8metric_report.json \
  >/tmp/qvg_cleanup_report_check.json

for f in outputs/qvgbench_cleanup_4prompts_seed42_genonly_20260509/grids/qvgcleanup_genonly_grid_prompt{0,1,2,3}.mp4; do
  test -s "$f"
done

python -m py_compile flashdreams/examples/qvg_benchmark.py
PYTHONPATH=flashdreams .venv/bin/python -m pytest flashdreams/tests/test_qvg_benchmark_metrics.py
```
