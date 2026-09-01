<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Upsample Video

Reusable v2 application infrastructure for playing an upscaled excerpt of
Blender's public 480p Big Buck Bunny video. Model integrations provide the
video post-processor and expose runnable entry-point slugs.

## Controls

None. The demo is uninteractive and stops after the configured number of
chunks.

## Usage

Launch through a model integration. For FlashVSR:

```bash
uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 upsample-video-flashvsr-v1.1-sparse-ratio-2.0 --output-path big-buck-bunny-upscaled.mp4 -- --max-chunks 4
```

Use <code>--mode webrtc</code> or <code>--mode native-window</code> instead of
<code>--output-path</code> to watch the run live.

Application arguments follow the final <code>--</code>:

| Argument | Default | Meaning |
| --- | ---: | --- |
| <code>--max-chunks</code> | <code>4</code> | Number of source-video chunks to process. |

Run <code>flashdreams-run-v2 &lt;upsample-video-slug&gt; -- --help</code> for
application help. The runtime's output and presentation arguments are
documented by <code>flashdreams-run-v2 --help</code>.

## Demo media attribution

[Big Buck Bunny](https://peach.blender.org/) is licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The demo downloads
and processes an excerpt from the 854x480 H.264 encode at runtime; the source
video is not redistributed in this repository.

> (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org

See the repository's `THIRD-PARTY-NOTICES` for the complete disclosure.

## Tests

~~~bash
uv run --package flashdreams-upsample-video-v2 --extra dev pytest apps/upsample_video -m ci_cpu -v
~~~

## Development

An integration constructs <code>UpsampleVideoApplication</code> with
<code>UpsampleVideoApplicationDefaults</code>, supplying its post-processor,
model name, and cold/steady input chunk sizes. Keep model-specific setup in
the integration adapter.
