# RUNTIME

[Options index](../OPTIONS.md)

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Total Blocks:** | `runtime.total_blocks` | Crazy Robotaxi | `--total-blocks` | Optional limit on generated model blocks; blank leaves the run unbounded. |
| **Prewarm Blocks:** | `runtime.prewarm_blocks` | Crazy Robotaxi | `--prewarm-blocks` | Blocks generated before play to warm the pipeline; must be nonnegative. |
