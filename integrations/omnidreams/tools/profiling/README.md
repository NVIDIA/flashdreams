# OmniDreams offline profiling � HW SM-cycles + SW throughput

Black-box, **offline** measurement of the token-streaming pipeline's per-chunk
cost. No live WebRTC session or browser needed � a random HDMap/latent stands in
for real conditioning, so the GPU work (kernels, shapes, dtypes) is identical to
what the server does per chunk.

The point: one small, self-contained set of scripts to answer
- **SW throughput** � how many frames/second the server generates (wall-clock,
  sync included), and
- **HW SM-cycle load** � how many SM cycles the GPU actually spends per chunk
  (measured with Nsight Compute, not `ms � clock`).

## Requirements
- The flashdreams venv (torch + `omnidreams` + the model weights cached via `HF_HOME`).
- For SM-cycles: **NVIDIA Nsight Compute** (`ncu`). GPU perf counters are
  admin-gated (`RmProfilingAdminOnly=1`), so run `ncu` as **root** (`sudo`).
  On a networked/`root_squash` home, point ncu's `HOME` at a locally-writable
  dir: `sudo -E env HOME=$NCU_HOME HF_HOME=$HF_HOME ncu ...`.

## 1. SW throughput  (`throughput_probe.py`)
Times a full generate loop over N chunks (cuda-graph/compile ON = production-like):
```bash
python throughput_probe.py --decode true  --chunks 300   # PIXEL  (encode+DiT+VAE-decode+finalize)
python throughput_probe.py --decode false --chunks 300   # TOKEN  (encode+DiT+finalize; decode skipped)
```
Prints `fps`, `ms/frame`, `chunks/s` over the measured window (after warmup).

## 2. HW SM-cycles  (`ncu_generate_probe.py`, `ncu_decode_probe.py`, `summarize_ncu.py`)
The probes bracket exactly the work to measure with `cudaProfilerStart/Stop`, so
`ncu --profile-from-start off` captures only that region (cuda-graph/compile are
turned OFF for clean per-kernel capture).

Full generate chunk (server), pixel vs token:
```bash
M=sm__cycles_active.sum,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed
ncu --profile-from-start off --metrics $M -f -o gen_pixel python ncu_generate_probe.py --decode true
ncu --profile-from-start off --metrics $M -f -o gen_token python ncu_generate_probe.py --decode false
```
VAE decode alone (also the **client-decode proxy** � same Blackwell GPU):
```bash
ncu --profile-from-start off --metrics $M -f -o decode python ncu_decode_probe.py
```
Summarize any report into headline numbers (SM active cycles, GPU time, throughput):
```bash
ncu --import gen_pixel.ncu-rep --csv --page raw > gen_pixel.csv
python summarize_ncu.py gen_pixel.csv
```

## Reading the results
- **token server cycles = pixel - decode**: token mode skips the server VAE
  decode (the client re-decodes), so the server sheds the decode cycles; the
  client picks them up (cost � the `ncu_decode_probe` number on the same GPU).
- Cycle counts are real; the **wall-time** printed by ncu is un-graphed (higher
  than production) � use `throughput_probe.py` (graphs ON) for real throughput.
