<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Omnidreams driving video

The first model on the v2 API that generates from something other than a
prompt. It continues from a first frame and follows an HDMap, the road layout,
for every frame it generates. That layout either comes from a recording of one
or is drawn as the run goes by the Ludus rasterizer. The model is the
single-view distilled checkpoint the `flashdreams-omnidreams` package already
configures for the v1 runner; this package is the application around it and
holds no model code of its own.

The Python module here is `omnidreams_v2` rather than `omnidreams`, which the
v1 package owns and this one imports.

## Set up

From the workspace root:

```bash
uv sync --package flashdreams-omnidreams-v2 --inexact
export HF_TOKEN=<your-hf-token>
```

The sync installs this application, the v1 `flashdreams-omnidreams` package
that holds the model and the Ludus rasterizer, and the `flashdreams-run-v2`
command itself. `--inexact` leaves anything else already in the environment
alone; drop it to have uv prune the environment down to what this needs. The
`interactive-drive` extra the v1 package documents is for its desktop
presenter, which nothing here uses.

The token is for Hugging Face, where the checkpoint, the sample recordings and
the scenes all live. The checkpoint is fetched on the first run, tens of
gigabytes of it including the Cosmos-Reason1 text encoder, so expect to wait.

## Generate a clip

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4
```

That downloads the default scene from `nvidia/omni-dreams-scenes` and has the
Ludus rasterizer draw the road layout while the run generates. Everything else
has a default too, including the frame the run continues from and the prompt,
both of which come out of the scene.

Writing to a file, the runner says nothing until it is done, when it prints the
path. So a run is silent through the download, the checkpoint, the rasterizer
build and the generating, and only the model's own output breaks that up. Start
with `--max-blocks 8 --no-compile`, below, to keep the silence short.

`uv run --no-sync` is what reaches the command, which the sync installed into
the workspace `.venv` rather than onto your `PATH`. Run
`source .venv/bin/activate` once instead and `flashdreams-run-v2` works on its
own.

Arguments after `--` go to the application, and `-- --help` lists them, though
the runner insists on an output path before it will get that far:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 -- --help
```

`--scene` drives a road other than the default, taking either an id to download
or a path to an archive of your own. This lists the ids the scenes dataset has:

```bash
uv run --no-sync python -c 'from omnidreams.scenes import list_available_scene_uuids; print("\n".join(list_available_scene_uuids()))'
```

Any of them goes straight after `--scene`:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --scene 0d404ff7-2b66-498c-b047-1ed8cded60d4
```

That particular id is the default, named explicitly.

## Where the road comes from

Drawing is the default because it is what a run that eventually steers needs: a
recording cannot show a road nobody drove down. Nothing steers yet, so a drawn
run follows the drive its scene recorded, which makes it as repeatable as
replaying a video of one. Once input is wired up, the poses stop coming from the
recording and start coming from what the driver did, and nothing above the
renderer changes.

A scene carries more than the layout. The frame the run continues from is the
one its front camera actually captured, and the drawing starts at that same
moment rather than at the top of the scene, so the model is shown the road it is
looking at. The scene's own description of the road becomes the prompt. The
recorded drive is sampled at 10Hz where the model generates at 30fps, so the
layout is resampled onto the generated rate. The default scene is 100 seconds of
road.

## Something to point the examples at

The examples below take a frame and a video of your own. If you have none to
hand, the bundled sample carries both, and this puts them in your shell:

```bash
eval "$(uv run --no-sync python -c 'from omnidreams_v2.samples import DEFAULT_HDMAP_SAMPLE, fetch_hdmap_sample; v, f = fetch_hdmap_sample(DEFAULT_HDMAP_SAMPLE); print(f"export HDMAP_VIDEO={v}"); print(f"export HDMAP_FRAME={f}")')"
echo "$HDMAP_VIDEO" "$HDMAP_FRAME"
```

It prints nothing itself -- `eval` eats what it printed, which is the point --
so the `echo` is how you see it worked. Give it a moment either way: importing
reaches the model package, and the files download the first time.

## Drive the same road differently

`--first-frame` and `--prompt` each replace what the scene supplied without
giving up the drawing, which is how one road is driven under weather it never
recorded:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --first-frame "$HDMAP_FRAME" --prompt "The same street after dark."
```

The layout is still drawn from the scene, and still drawn from the moment the
scene recorded, since a frame of your own says nothing about where along the road
it was taken. Only the picture the model continues from changes. Naming a frame
is not asking to replay anything, so the default scene keeps being drawn when
`--scene` says nothing -- as above.

The two are worth changing together. A first frame showing a different road than
the layout describes hands the model a contradiction, and a prompt still
describing the scene's own weather works against a frame that shows other
weather. The sample's frame above is a different road from the default scene, so
that command shows the path working rather than showing it working well; a frame
of the same road under other conditions is what this is for.

The first drawn run pauses a few minutes to build the rasterizer's CUDA
extension, which needs `nvcc` new enough for your GPU on `PATH` or at
`CUDA_HOME` -- 12.8 or later for Blackwell. The build failing with `unsupported
gpu architecture` means the toolkit is older than the card.

## Replay a recording instead

`--hdmap` gives up the rasterizer and reads the layout from video someone
already rendered, which is what a benchmark wants and what runs on a machine
with no scene and no rasterizer:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --hdmap
```

Bare, that fetches the default recording from the `nvidia/omni-dreams-samples`
dataset. A sample carries the frame to continue from as well as the layout, so
it needs nothing else said about it. Naming an id picks another, and they are
listed
[here](https://huggingface.co/datasets/nvidia/omni-dreams-samples/tree/main/data/single_view):

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --hdmap 239560dc-33d1-11ef-9720-00044bcbccac
```

That one is the default again, named rather than left unsaid.

Point `--hdmap` at files instead to replay a recording of your own, one video
per camera, alongside the frame each continues from:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --hdmap "$HDMAP_VIDEO" --first-frame "$HDMAP_FRAME"
```

Which of the two you meant is read off the file extension, the way the samples
are named: `--hdmap 239560dc-...` is an id to download and `--hdmap road.mp4` is
a file to read, so a misspelled path is reported as the missing file it is.

Naming a recording and a scene at once is refused rather than resolved, since
they are two answers to the one question of where the layout comes from.
`--first-frame` is not one of those answers, which is why it works either way:
it says what the run continues from, and a drawn run answers that too.

## How long a run is

About a minute, which is what a run that was told nothing produces. Long enough
to see whether a drive holds together and short enough to wait for, since
generating is real time at best.

`--max-blocks` says otherwise. `--max-blocks 8` is 61 frames, about two
seconds, which is what you want for a first run; pair it with `--no-compile`,
since compilation is on in the model's own config and costs minutes on the first
run to save milliseconds a block.

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --max-blocks 8 --no-compile
```

`--max-blocks 0` drives to the end of the road however long that takes, which is
what an interactive session wants and what the default minute would otherwise
cut off. On the default scene that is 100 seconds of video, so it is a longer
wait than anything else here:

```bash
uv run --no-sync flashdreams-run-v2 omnidreams --output-path drive.mp4 \
    -- --max-blocks 0
```

Either way the road can end first, and the run stops on a block boundary rather
than on a block it has only part of the layout for. The first block decodes 5
frames and every block after it 8, at 30 frames per second, so a ten-second
recording is about 38 blocks and a minute is 226.

## What it generates

1280x704 frames at 30fps, laid out `bvtchw`, as `[-1, 1]` floats on the GPU.
Those numbers are the checkpoint's, read off the runner config the
`flashdreams-omnidreams` package ships rather than written down here. Something
else can be asked for with `--pixel-width` and `--pixel-height` before the `--`,
each a multiple of 8.

One camera per run: an MP4 holds one sequence of frames, so the file window
rejects output with more than one view in it. The multi-view checkpoints need a
window that lays the cameras out, which is not built yet.

## The seams underneath

`HDMapSource` in `conditioning.py` is the seam a session reads its conditioning
through, and has two implementations. `RenderedHDMapSource` keeps a run's place
along a scene and draws each chunk as the run reaches it.
`PrecomputedHDMapSource` reads the same chunks out of recorded video instead.
A session cannot tell which it was given.

Under the drawn one is a smaller seam, `SceneRenderer`, whose only
implementation is `LudusSceneRenderer` in `ludus.py`. That split is what keeps
the CUDA rasterizer out of the tests: everything around it -- working along a
scene consecutively, turning bytes into the pixels the model reads, starting the
layout at the moment the run continues from -- is covered on a CPU against a
stand-in renderer, and each of those would produce a plausible-looking wrong
drive rather than an error.

Steering is what this shape is for. `ISession.step` already receives input
events and passes them to the source, where a drawn source is the one that could
act on them; today it ignores them and follows the recorded drive.

## Tests

```bash
uv sync --package flashdreams-omnidreams-v2 --group test --inexact
uv run --no-sync pytest integrations_v2/omnidreams -m ci_cpu -v
```

Those run the application against a stand-in model, a stand-in drive and a
stand-in renderer, which covers what is particular here: each block is
conditioned on exactly the frames it generates, a run ends when the road does,
and a drawn run is shown the road from the moment it continues from.
