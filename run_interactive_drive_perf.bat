@echo off
setlocal enableextensions enabledelayedexpansion

REM ==========================================================================
REM PERF variant of run_interactive_drive.bat: launches interactive-drive with
REM the perf-tuned manifest (example_world_model_perf.yaml) for higher FPS:
REM   - lower render res (1168x640), denoising_steps [1000, 100], compile_net
REM   - native_dit_acceleration: auto -> tries the single-view FP8 DiT ext and
REM     FALLS BACK to PyTorch if it can't build on Windows (ext not prebuilt).
REM First launch is SLOWER (torch.compile warmup + Ludus JIT); caches persist
REM in-repo so later launches are fast. For true FP8 the native ext must build
REM (see the OmniDreams single-view Windows build recipe), then set the manifest
REM back to native_dit_acceleration: required to force-verify FP8.
REM   run_interactive_drive_perf.bat            perf minimap + world model
REM   run_interactive_drive_perf.bat --no-hud   pass any demo args through
REM ==========================================================================

cd /d C:\workspace\world\flashdream_public

set "VENV=C:\workspace\world\flashdream_public\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: flashdream .venv not found at %VENV% & exit /b 1 )

REM .venv\Scripts on PATH so torch's JIT finds ninja.exe (+ rerun.exe).
set "PATH=%VENV%\Scripts;%PATH%"

REM DO NOT call vcvars64 here. The Ludus torch C++/CUDA extension AND triton-windows
REM each run their OWN MSVC detection (setuptools _get_vc_env) at compile time. Pre-running
REM vcvars64 makes theirs a SECOND vcvars pass, which corrupts the Windows SDK ucrt include
REM into a space-stripped "C:\Program Files(x86)\...\ucrt" (doesn't exist) -> cl can't find
REM <malloc.h> -> `alloca` unresolved -> LNK1120 in the Triton JIT (torch._inductor).
REM Verified on this box: no-vcvars compiles clean; vcvars64-then-triton fails every time.
REM So leave the compiler env to the tools; only set CUDA below (nvcc needs it, not from vcvars).
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%PATH%"
REM RTX 5090 (sm_120): force the arch for any torch JIT (overrides stale machine value).
set "TORCH_CUDA_ARCH_LIST=12.0"

REM Windows SDK include paths for MSVC cl.exe (windows.h, assert.h, etc).
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;%LIB%"

REM PhysX runtime DLLs and Visual C++ runtime
set "PATH=C:\Users\kschmid\AppData\Local\ludus-renderer\physx-5.9.0\build-windows-AMD64\physx-lib\bin\win.x86_64.vc143.md\release;%PATH%"
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\x64\Microsoft.VC143.CRT;%PATH%"

REM Disable HuggingFace symlink checking (Windows permission issue on .gitattributes)
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM HF token from the cached token file if not already set.
if "%HF_TOKEN%"=="" if exist "C:\Users\kschmid\.cache\omni-dreams\huggingface\token" set /p HF_TOKEN=<"C:\Users\kschmid\.cache\omni-dreams\huggingface\token"

REM Inductor: ATen backends only (avoids the lightVAE Triton >99KB-smem OOM crash),
REM no autotune sweep, and PERSISTENT compile caches in-repo (not %TEMP%, which gets
REM cleaned and forces a full recompile every launch).
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN"
set "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN"
set "TORCHINDUCTOR_MAX_AUTOTUNE=0"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0"
set "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
set "TORCHINDUCTOR_CACHE_DIR=%~dp0.cache\torchinductor"
set "TRITON_CACHE_DIR=%~dp0.cache\triton"
set "TORCHINDUCTOR_COMPILE_THREADS=1"
if not exist "%~dp0.cache" mkdir "%~dp0.cache"

REM 32GB GPU vs ~48GB nominal: cut VRAM fragmentation.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM Strip inherited venv state so the venv loads its own stdlib cleanly.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"

REM Enable debug logging
set "LOGLEVEL=DEBUG"
set "PYTHONUNBUFFERED=1"
set "LOGURU_LEVEL=DEBUG"

REM Perf-tuned manifest (compile_net:true, low-res, few-step, native auto).
set "MANIFEST=C:\workspace\world\flashdream_public\integrations\omnidreams\omnidreams\interactive_drive\configs\example_world_model_perf.yaml"

REM HUD goal-marker / cuboid knobs (same as the base launcher).
set "IDRIVE_TEST_MARKER_AHEAD_M=50"
if not defined IDRIVE_ROAD_CUBOIDS_AHEAD set "IDRIVE_ROAD_CUBOIDS_AHEAD="
set "IDRIVE_DEBUG_ZONES=1"
set "IDRIVE_LOG_FILE=C:\tmp\idrive_perf.log"
if not exist "C:\tmp" mkdir "C:\tmp"

echo.
echo ===================================================================
echo LAUNCHING INTERACTIVE-DRIVE PERF WITH PHYSICS
echo ===================================================================
echo Manifest: %MANIFEST%
echo Game mode: ENABLED ^(collisions + physics^)
echo Offload text encoder: enabled
echo Resolution: 1168x640 (perf tuned)
echo Denoising steps: [1000, 100]
echo Native acceleration: auto-fallback to PyTorch
echo ===================================================================
echo Controls: WASD=drive Mouse=look C=obstacle R=restart Esc=quit
echo ===================================================================
echo.

REM Overview minimap: fixed map-centre camera; --bev-fov-deg used for the fit,
REM --bev-height-m / --bev-tilt-deg ignored in overview. --no-bev-overview for
REM the old ego-centred/heading-up minimap.
echo [INIT] Starting event loop...
"%VENV%\Scripts\interactive-drive.exe" --manifest "%MANIFEST%" --offload-text-encoder --bev-tilt-deg 0 --bev-height-m 1200 --bev-fov-deg 60 --game-mode %*
echo [EXIT] interactive-drive closed
set EXIT_CODE=%ERRORLEVEL%

if not %EXIT_CODE%==0 ( echo. & echo interactive-drive exited with code %EXIT_CODE% & exit /b %EXIT_CODE% )
endlocal
