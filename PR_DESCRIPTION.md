## Summary

- Add persistent, prewarmed postprocessing for the Lingbot Cam2V rollout.
- Let any configured postprocessor declare its output size; the app presents that size without mutating the model session descriptor.
- Add `--postprocess-comparison-ui` to show the raw/upscaled and postprocessed streams side by side.
- Keep postprocessing running when its UI toggle changes, and select only which output frames are presented.
- Retain the existing in-flight reset path while clearing postprocessor temporal state in place; no restart UI is added.
- Preserve presentation backlog draining and scope the change away from Interactive Drive.

## Validation

- Added CPU coverage for postprocessor streaming, output selection, comparison composition, and presentation pacing.
- Full WebRTC/GPU rollout remains the recommended end-to-end validation.
