---
name: flashdreams-preparation
description: Use when changing FlashDreams V2 downloads, compilation, native loading, model construction, static validation, or application preloading.
---

# FlashDreams preparation

## Rules

- Start downloads, extraction, compilation, native loading, and model construction in `IApplication.init` or nested within.
- Logic after `IApplication.init` should consume a cached 'prepared' state. If data is missing, raise an error; do not prepare lazily.
- Do not add runtime wrappers merely to hack around and silence static analysis.

## Commands

Static analysis on "will preload produce a complete cache"; does not call `IApplication.init`:

```bash
uv run flashdreams-run-v2 <slug> --preload-application validate
```

Static analysis followed by `IApplication.init`; does not create a session:

```bash
uv run flashdreams-run-v2 <slug> --preload-application full -- <app-args>
```

## Static analysis

The analyzer checks application modules for calls to:

- `subprocess.Popen`
- `huggingface_hub.hf_hub_download`
- `huggingface_hub.snapshot_download`
- `urllib.request.urlopen`
- `urllib.request.urlretrieve`

It scans modules loaded while constructing the application plus the application's defining module.

Move internal findings into `IApplication.init`. External findings may be unreachable by user; use the reported internal call-site trace to decide next steps.
