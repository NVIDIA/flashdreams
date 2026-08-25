# Waypoint 1.5 V2 application

This package adapts the independently authored `flashdreams-waypoint` model
package to FlashDreams' V2 application, session, model-loop, event, and output
APIs. Model modules are loaded once per application; the image-established
transformer/decoder cache, RNG stream, and live controls are isolated per
session.

The application always declares four-frame `TCHW` results at 1280x720 and
60 FPS playback. Waypoint generates on its native 1024x512 canvas and the
adapter resizes results for presentation. File controls use blocking,
new-results-only presentation so MP4 output retains every generated frame.

See [ADR-1-control-events.md](ADR-1-control-events.md) for the live keyboard and
mouse contract. Passing `--controls-file` selects finite deterministic mode;
omitting it selects live input. `--example-data` uses the pinned public seed
and bundled 118-action timeline.
