<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2V

Reusable v2 video-to-video application infrastructure. It upscales a selected
local or remote video and writes or streams the resulting video. When no input
is selected, it uses an excerpt of Blender's public 480p Big Buck Bunny video.
Model integrations provide the video post-processor and expose runnable
entry-point slugs.

## Controls

None. The demo is uninteractive and stops after the configured number of
chunks.

## Usage

Launch through a model integration. For FlashVSR:

```bash
uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 v2v-flashvsr-v1.1-sparse-ratio-2.0 --output-path upscaled.mp4 -- --video-path input.mp4
```

Omit `--video-path` to download and process the bounded Big Buck
Bunny default used by the original demo. Input paths may also be HTTP(S) URLs.

Use `--mode webrtc` or `--mode native-window` instead of
`--output-path` to watch the run live.

Application arguments follow the final `--`:

| Argument | Default | Meaning |
| --- | ---: | --- |
| --video-path | Big Buck Bunny | Local video path or HTTP(S) URL. |
| --max-chunks | all selected-video chunks; 100 for Big Buck Bunny | Maximum number of source-video chunks to process. |

Run `flashdreams-run-v2 <v2v-slug> -- --help` for
application help. The runtime's output and presentation arguments are
documented by `flashdreams-run-v2 --help`.

## Demo media attribution

[Big Buck Bunny](https://peach.blender.org/) is licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The demo downloads
and processes an excerpt from the 854x480 H.264 encode at runtime; the source
video is not redistributed in this repository.

> (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org

See the repository's `THIRD-PARTY-NOTICES` for the complete disclosure.

## Tests

~~~bash
uv run --package flashdreams-v2v --extra dev pytest apps/v2v -m ci_cpu -v
~~~

## Development

An integration constructs `V2VApplication` with
`V2VApplicationDefaults`, supplying its post-processor,
model name, and cold/steady input chunk sizes. The application resolves output
dimensions and frame rate from the selected input. Keep model-specific setup
in the integration adapter.
