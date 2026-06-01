# GB200 OmniDreams WebRTC Perf Repro

This note captures the command used to reproduce and profile the OmniDreams
WebRTC serving path on a GB200/GB300 host. It runs the issue-195 renderer
profiling branch with Ludus render profiling enabled and writes per-chunk timing
records to `issue195_webrtc_profile_latest.log`.

## Checkout

```bash
cd ~/github/flashdreams
git fetch origin refs/pull/215/head:refs/remotes/origin/pull-request/215
git worktree add -B issue-195-render-profile-repro \
  ~/github/flashdreams-issue-195-render-profile-repro \
  origin/pull-request/215
cd ~/github/flashdreams-issue-195-render-profile-repro
```

## Environment

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive

# Keep cuDNN sublibraries aligned with CUDA 13.2 system images. Without this,
# cuDNN 9.20 can load the host's 9.21 tensor-IR engine and fail with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
uv pip install 'nvidia-cudnn-cu13==9.21.1.3'
```

## Launch Profiled Server

Use the Hugging Face scene path when the scene asset is available:

```bash
export OMNIDREAMS_LUDUS_RENDER_PROFILE=1
export OMNIDREAMS_LUDUS_RENDER_PROFILE_CUDA_EVENTS=1
unset OMNIDREAMS_LUDUS_MSAA_SAMPLES
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 uv run --no-sync --package flashdreams-omnidreams \
  torchrun --nproc_per_node 1 -m omnidreams.webrtc.server \
  --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
  --scene-uuid 065dcac9-ee67-4434-a835-c6b816c88e48 \
  --port 8099 \
  2>&1 | tee issue195_webrtc_profile_latest.log
```

If the Hugging Face scene URL returns 404 but the scene is already staged
locally, launch from the cached scene directory instead:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync --package flashdreams-omnidreams \
  torchrun --nproc_per_node 1 -m omnidreams.webrtc.server \
  --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
  --scene_dir ~/.cache/flashdreams/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48 \
  --port 8099 \
  2>&1 | tee issue195_webrtc_profile_latest.log
```

Open the viewer after startup:

```text
http://<server-ip>:8099/request_session
```

Drive for 20-30 seconds. The first chunks can include compile and autotune
noise, so summarize later chunks.

## Summarize

```bash
uv run --no-sync python integrations/omnidreams/scripts/summarize_webrtc_profile.py \
  issue195_webrtc_profile_latest.log \
  --min-chunk 4
```

The issue-195 HSG signature is renderer-bound serving:

```text
wrapper_render_condition_ms ~= renderer_ctx_render_ms
renderer_ctx_render_ms      ~= ctx_render_plugin_cuda_ms_sum
```

On the local GB300 loopback run, stable chunks 4-5 showed renderer context
rendering around 15-20 ms and did not reproduce the original ~260 ms renderer
bottleneck.
