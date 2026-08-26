# MiniMax H3 V2 applications

This package exposes three finite, synchronized video-and-audio applications:

- `minimax-h3-t2va` for prompt-only generation;
- `minimax-h3-fl2va` for a first keyframe, last keyframe, or both;
- `minimax-h3-ref2va` for ordered image, video, and audio references.

The runtime owns presentation and output targets. The application decodes local
input media, stages one native H3 component at a time, and returns a TCHW video
tensor plus stereo 32 kHz normalized PCM. Production inference does not import
Diffusers or PyAV.

Application arguments go after the V2 command's `--` separator. Output geometry
and the fixed 24 fps rate remain runtime arguments before that separator:

```bash
flashdreams-run-v2 minimax-h3-t2va \
  --pixel-width 768 --pixel-height 768 --fps 24 \
  --mode mp4 --output-path result.mp4 -- \
  --prompt "A lantern floating through a rainy night market" \
  --duration 5 --steps 30 --seed 42
```

FL2VA accepts `--image-path` and `--last-image-path`. REF2VA accepts repeated
ordered specifications such as `--reference image:subject.png`,
`--reference video:motion.mp4`, and `--reference audio:voice.wav`.

Pass both `--work-dir` and `--job-id` to checkpoint paired video/audio denoise
state. `--restart` ignores a matching checkpoint. `--lora` currently accepts a
local Musubi safetensors file whose contents become part of checkpoint identity.

Still images are decoded with Pillow. Video and audio use executables named
`ffprobe` and `ffmpeg` installed on the host and available on `PATH`; the
package does not bundle or discover a Python-provided FFmpeg.

The built-in MP4 mode publishes synchronized audio as AAC-LC at 192 kbit/s.
Before creating output staging or beginning model generation it asks the host
`ffmpeg` executable to encode one silent frame using the session's exact
sample rate and channel count. Video and normalized `f32le` PCM remain private
until the completed streams are muxed and atomically replace the requested
output. Audio is padded or trimmed to the exact written-video timeline rather
than being silently dropped or allowed to drift.

See [ROLLOUT.md](ROLLOUT.md) for pinned revisions, CPU/CUDA gates, real-weight
RTX PRO results, and external `ffmpeg`/`ffprobe` validation.
