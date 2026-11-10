---
name: flashdreams-package-offline
description: Package an installed FlashDreams V2 application as an offline executable folder and diagnose preload or bundled-cache failures. Use for native-window or WebRTC distribution without Python, uv, or network access on the target. Ordinary demo launching, generic Python packaging, and website builds are outside this workflow.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Package FlashDreams for offline use

Use the user's checkout at the requested revision. For plans or command reviews,
inspect docs and source without downloading assets or building a bundle.

1. Follow `docs/source/documentation/tools/offline_program_packager.md`
   ([offline packager](https://flashdreams.org/main/documentation/tools/offline_program_packager/)).
   Confirm a working documented V2 preset, its dependencies and the target
   OS/GPU/driver. Try to output in a new directory for every new package.
2. Inspect `INSTALLER_OUTPUT.txt` and any `PREPARATION_ISSUES.txt` to help
   identify/fix late downloads/builds in application initialization. Fixing
   reported issues is not mandatory and does not guarantee the app will function.
3. Report the exact command, revision, target, artifacts and checks performed.
   Distinguish command validation, a successful build and verified offline use.
