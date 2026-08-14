# FlashDreams V2V application

`flashdreams-v2v` provides transport-neutral application and session primitives
for file-backed video-to-video inference. Model integrations own concrete
applications and register them through the `flashdreams.applications` entry
point group.

The reusable application accepts `--input-path`, optional `--device`, and
optional output `--fps`. Its session loads the source video, plans a cold-start
chunk followed by steady-state chunks, and feeds each normalized `[B, C, T, H,
W]` chunk to the integration pipeline.
