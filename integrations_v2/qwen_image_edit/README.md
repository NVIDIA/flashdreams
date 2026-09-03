# Qwen Image Edit V2

Native implementation of Qwen Image Edit 2511 and a one-shot FlashDreams V2
application. PyTorch implements the diffusion transformer and Qwen image VAE;
Transformers supplies the Qwen2.5-VL prompt/image encoder. Diffusers is used
only by the CPU parity test.

The checkpoint is pinned to Hugging Face revision
`6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`. The default follows the official
40-step, true-CFG 4.0 sampling path, moving the vision encoder, transformer, and
VAE onto the GPU one at a time.

```bash
uv run --package flashdreams-qwen-image-edit-v2 qwen-image-edit-v2 \
  --mode png --output-path output.png --width 1280 --height 704 -- \
  --input semantic-road.png --prompt "Turn this into a daylight city street." \
  --negative-prompt "cars, traffic" --true-cfg-scale 4
```

The application emits one TCHW frame and finishes, so it works with PNG, MP4,
WebRTC, and native-window V2 presentation modes. PNG is the natural mode for
first-frame authoring.
