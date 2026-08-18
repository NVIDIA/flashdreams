# Sub-task 1a — streaming cache-as-IO export + multi-chunk continuity (throwaway)

Not product code. Proves the hardest remaining piece before productionizing the
client decoder: **can we export the TAEHV decoder with its temporal cache as
explicit ONNX inputs/outputs, and does threading that cache reproduce the native
streaming decode across many chunks (no drift)?** Delete before any PR.

The decoder keeps a small per-block "last-frame" cache (9 slots). The export
turns that internal `Dict[id(module)]` cache into ordered tensor I/O
(`cache_in_*` → `cache_out_*`) so the browser can carry it chunk-to-chunk.

## Commands (GPU box, repo uv env)

```bash
# 1. Export the steady-state, cache-as-IO ONNX
uv run python spike/1a-streaming-cache/export_streaming.py

# 2. Validate multi-chunk continuity vs the native torch streaming decoder
uv run python spike/1a-streaming-cache/validate_continuity.py
```

`validate_continuity.py` streams N=6 consecutive chunks and prints per-chunk PSNR
for two checks, both seeded from the native decoder's post-chunk-0 cache and then
**threading their own cache forward**:

- **[A] wrapper (torch)** — the cache-as-IO dict↔list logic. *Expect ~exact:* very
  high PSNR (>100 dB) that stays flat across chunks. Anything lower means the
  cache reconstruction/extraction is wrong.
- **[B] exported ONNX (onnxruntime)** — the exported graph. *Expect high and
  non-drifting* PSNR across all chunks (small, constant export/runtime delta).
  If `onnxruntime` isn't installed it's skipped with a note — [A] already proves
  the cache logic; install with `uv pip install onnxruntime-gpu` for [B].

## Reading the result

- **[A] flat, ~exact + [B] high and non-drifting across all 6 chunks** →
  cache-as-IO export is faithful and continuity holds → proceed to productionize
  the client decoder around this export.
- **PSNR that degrades chunk-over-chunk** → cache drift; something in the
  threading/extraction is off — paste the numbers and I'll fix.

## Likely iteration points (I can't run these here)

- `torch.onnx.export` with the 10-in/10-out signature and the in-graph dict
  reconstruction may warn (e.g. about the `MemBlock` in-place cache write); a
  warning is fine, a hard error we fix.
- Module path `decoder.taehv.decoder.blocks` / `MemBlock` / `TAEHVCache` import —
  if any name is off you'll get an ImportError/AttributeError; paste it.
