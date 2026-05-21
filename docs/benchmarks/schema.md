# FlashDreams benchmark schema

`benchmark_results.json` contains a top-level object:

- `schema_version`: semantic version for schema compatibility.
- `updated_at`: ISO-8601 timestamp.
- `records`: array of benchmark records.

Each record includes:

- `benchmark_id` (`string`): unique id for the measurement row.
- `workload` (`string`): workload name (for example `self_forcing_block6`).
- `model` (`string`): model family and variant.
- `hardware` (`string`): hardware target (`H100`, `GB200`, `GB300`, ...).
- `parallelism` (`string`): launch mode (`1xGPU`, `4xGPU`, ...).
- `method` (`string`): implementation being compared (`flashdreams`, `official`,
  `fastvideo`, `lightx2v`).
- `status` (`string`): one of `pass`, `oom`, or `missing`.
- `metrics` (`object`): numeric metrics in milliseconds/GiB where available.
  - `total_ms` (`number | null`)
  - `dit_ms` (`number | null`)
  - `vae_ms` (`number | null`)
  - `kv_update_ms` (`number | null`)
  - `gpu_mem_gib` (`number | null`)
- `provenance` (`object`): reproducibility context.
  - `command` (`string | null`)
  - `commit` (`string | null`)
  - `notes` (`string`)

Records with `status=missing` are placeholders to reserve comparison slots for
future results and are ignored by chart generation.
