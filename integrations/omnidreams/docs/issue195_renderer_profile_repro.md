# Issue 195 Renderer Profiling Repro

This branch includes temporary profiling instrumentation for OmniDreams WebRTC
serving. It records whether end-to-end serving time is spent in the model
pipeline, queue/enqueue logic, or Ludus render conditioning.

## Base

The original repro was run from:

```text
upstream/main
3816e32d99a001eab91996f1e52fe488be1ee9cf
```

Use this branch directly after it is published. No patch application is needed.

## Launch On HSG

Set `HF_TOKEN` or `HF_TOKEN_FILE` first. Then request a full GB200 node, but run
the WebRTC server with one visible GPU, matching the OmniDreams WebRTC README
path.

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/flashdreams
git fetch <remote> issue-195-render-profile-repro
git switch issue-195-render-profile-repro

uv sync --package flashdreams-omnidreams --extra interactive-drive

srun -A nvr_torontoai_videogen -p batch --qos=interactive \
  -N1 --gpus-per-node=4 --ntasks=1 --cpus-per-task=16 --time=02:00:00 \
  --chdir "$PWD" \
  bash -lc '
    export OMNIDREAMS_LUDUS_RENDER_PROFILE=1
    export OMNIDREAMS_LUDUS_RENDER_PROFILE_CUDA_EVENTS=1
    unset OMNIDREAMS_LUDUS_MSAA_SAMPLES
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync --package flashdreams-omnidreams \
      torchrun --nproc_per_node 1 -m omnidreams.webrtc.server \
      --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
      --scene-uuid 065dcac9-ee67-4434-a835-c6b816c88e48 \
      --port 8095 2>&1 | tee issue195_webrtc_profile.log
  '
```

`OMNIDREAMS_LUDUS_MSAA_SAMPLES` is intentionally unset so the renderer uses its
default `4` samples. The profile log should include `renderer_msaa_samples: 4.0`.

## Forward And Drive The Viewer

In another shell, replace `<allocated-node>` with the Slurm node name:

```bash
socat -d -d TCP-LISTEN:8097,reuseaddr,fork,bind=0.0.0.0 \
  TCP:<allocated-node>.cm.cluster:8095
```

Open:

```text
http://oci-hsg-cs-001-vscode-01:8097/request_session
```

Click Connect Session and drive for 20-30 seconds. The first few chunks may
include warmup or compile noise, so summarize later chunks.

## Summarize

```bash
python integrations/omnidreams/scripts/summarize_webrtc_profile.py \
  issue195_webrtc_profile.log \
  --min-chunk 4
```

Use `--session latest` for the most recent browser run, or `--session all` to
include loopback warmup and browser sessions together.

## Expected Signature

The model pipeline is fast enough for more than 40 FPS, but the end-to-end
viewer generation loop is around 18-20 FPS because render conditioning dominates.

Observed browser-session summary from the original repro:

```text
selected_chunks=31 session=latest min_chunk=4
gen_ms                             avg=  439.3 ms / 8 frames
wrapper_render_condition_ms        avg=  259.8 ms
renderer_ctx_render_ms             avg=  258.6 ms
ctx_render_plugin_cuda_ms_sum      avg=  254.9 ms
pipeline_total_ms                  avg=  175.0 ms
pipeline_total_ms_wo_finalize      avg=  126.2 ms
enqueue_ms                         avg=   26.4 ms
```

Interpretation:

```text
wrapper_render_condition_ms ~= renderer_ctx_render_ms
renderer_ctx_render_ms      ~= ctx_render_plugin_cuda_ms_sum
```

So the extra serving time is inside `self.ctx.render(...)`, and nearly all of
that time is accumulated by Ludus CUDA render plugin calls across the 8
conditioning frames.
