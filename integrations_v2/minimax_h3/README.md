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

Image and video pixels are decoded by the shared runner media boundary. Audio
uses executables named `ffprobe` and `ffmpeg` installed on the host and available
on `PATH`; the package does not bundle or discover a Python-provided FFmpeg.

The application already declares and returns synchronized audio. Publishing it
in the built-in MP4 mode remains gated on an explicitly approved public MP4
audio codec; the existing video-only sink rejects audio rather than silently
dropping it.

See [ROLLOUT.md](ROLLOUT.md) for pinned revisions, CPU/CUDA gates, real-weight
RTX PRO results, and the remaining MP4-audio approval boundary.
