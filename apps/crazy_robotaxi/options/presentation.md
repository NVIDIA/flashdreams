# PRESENTATION

[Options index](../OPTIONS.md)

These fields apply when saved, without a process restart, unless a command-line override is active.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Width:** | `presentation.width` | Crazy Robotaxi | `--display-width` | Display width in pixels; blank uses the model width. Set with Height. |
| **Height:** | `presentation.height` | Crazy Robotaxi | `--display-height` | Display height in pixels; blank uses the model height. Set with Width. |
| **Hud Enabled:** | `presentation.hud_enabled` | Crazy Robotaxi | — | Shows the gameplay HUD. |
| **Show Fps:** | `presentation.show_fps` | Crazy Robotaxi | `--show-fps`, `--no-show-fps` | Shows the frame-rate counter. |
| **Show Current Prompt:** | `presentation.show_current_prompt` | Crazy Robotaxi | — | Shows the world-model prompt. |
| **Show Control Hints:** | `presentation.show_control_hints` | Crazy Robotaxi | — | Shows the control help on the HUD. |
| **Show Live Edit Buttons:** | `presentation.show_live_edit_buttons` | Crazy Robotaxi | — | Shows live-edit ability buttons. |
| **Live Edit Mapping Location:** | `presentation.live_edit_mapping_location` | Crazy Robotaxi | — | Places live-edit mappings in `buttons` or `control hints`. |
