---
name: flashdreams-package-offline
description: Package an installed FlashDreams V2 application as an offline executable folder and diagnose preload or bundled-cache failures. Use for native-window or WebRTC distribution without Python, uv, or network access on the target. Ordinary demo launching, generic Python packaging, and website builds are outside this workflow.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Package FlashDreams for offline use

Use the user's checkout and requested revision. For plans or command-only tasks,
read the guide and tool source without starting downloads or a bundle build.

1. Follow `docs/source/documentation/tools/offline_program_packager.md`
   ([offline packager](https://flashdreams.org/main/documentation/tools/offline_program_packager/)).
   Confirm a working documented V2 preset, its dependencies and the target
   OS/GPU/driver. The packager supports `native-window` and `webrtc`; build on
   the target OS rather than assuming cross-platform binaries.
2. Check the tool's current help and the
   [CLI reference](https://flashdreams.org/main/documentation/cli/). Put packager
   options before `:::`, then the full `flashdreams-run-v2` command. Its runtime
   arguments precede `--`; app arguments follow it. Use a new output directory.
3. Let the packager control preload. Inspect `INSTALLER_OUTPUT.txt` and
   `PREPARATION_ISSUES.txt`; fix late downloads/builds in application initialization
   rather than relying on a developer cache. Skipping runtime validation does
   not skip preparation or prove that the application works.
4. Follow the guide's bundle-content and writable-cache instructions. Distribute
   the complete folder, and check model/asset/dependency redistribution terms.
   Test on a clean compatible target with networking disabled and empty user
   caches; verify actual generation, restart and the requested output mode.
5. Report the exact command, revision, target, artifacts and checks performed.
   Distinguish command validation, a successful build and verified offline use.
