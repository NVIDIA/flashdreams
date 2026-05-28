# Assets

Helper code only -- this directory holds the unpacked-scene-bundle
loader (`scene_bundle.py`). Scene USDZs themselves are staged into the
shared `omnidreams` scene cache under
`$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/`, **not** here. See
`omnidreams.scenes` and `omnidreams-prepare` for how staging
works; both the desktop demo and the WebRTC server consume from the
same cache root.
