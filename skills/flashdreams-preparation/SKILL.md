---
name: flashdreams-preparation
description: Use when changing FlashDreams V2 downloads, compilation, native loading, model construction, runtime validation, or application preloading.
---

# FlashDreams preparation

## Rules

- Start downloads, extraction, compilation, native loading, and model construction in `IApplication.init` or nested within.
- Logic after `IApplication.init` should consume a cached 'prepared' state. If data is missing, raise an error; do not prepare lazily.

## Commands

```bash
uv run flashdreams-run-v2 <slug> --preload-application -- <app-args>
```

Preload runs `IApplication.init` and one model block. It defaults to `warn`;
ordinary runs default to `none`. Override either with
`FLASHDREAMS_PREPARATION_POLICY=none|warn|error`.

Findings include a full stack trace with every detected "failure", emitting results in
`preparation_issues.txt`, or `FLASHDREAMS_PREPARATION_ISSUES_PATH` when set.

All reported work should be moved into `IApplication.init` if operation is a one-time download/setup.
Packager records validation in `INSTALLER_OUTPUT.txt`

`--skip-preload-validation` runs preloading (`init`) but omits the runtime validation of running the first model block.
