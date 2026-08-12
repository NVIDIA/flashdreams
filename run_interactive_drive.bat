@echo off
setlocal enableextensions enabledelayedexpansion

REM ==========================================================================
REM Launch the omnidreams interactive-drive desktop demo in flashdream's .venv,
REM with the full Windows build env the Ludus HD-map renderer needs (it
REM JIT-compiles a CUDA/C++ torch extension on first launch).
REM   run_interactive_drive.bat            no auto-cubes; press 'c' to drop one
REM   run_interactive_drive.bat --no-hud   pass any demo args through
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
set "PATH=%CUDA_HOME%\bin;%PATH%"
REM RTX 5090 (sm_120): force the arch for any torch JIT (overrides stale machine value).
set "TORCH_CUDA_ARCH_LIST=12.0a"

REM Windows SDK ucrt include path for MSVC cl.exe (assert.h not found fix).
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;%LIB%"

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

REM Eager low-res manifest (compile_net:false) for fast GUI bring-up.
set "MANIFEST=C:\workspace\world\flashdream_public\integrations\omnidreams\omnidreams\interactive_drive\configs\example_world_model.yaml"

REM HUD goal-marker / cuboid knobs. Empty cuboids = none at launch; press 'c' in
REM the demo to drop an obstacle cuboid ~14 m ahead of the car on demand.
set "IDRIVE_TEST_MARKER_AHEAD_M=50"
set "IDRIVE_ROAD_CUBOIDS_AHEAD="
REM Debug render of the box zones: draws the START (green) + TARGET (blue)
REM wireframe cubes in the main view and the BEV minimap. Set empty to disable.
set "IDRIVE_DEBUG_ZONES=1"
set "IDRIVE_LOG_FILE=C:\tmp\idrive.log"
if not exist "C:\tmp" mkdir "C:\tmp"

echo Launching interactive-drive ( args: %* )
REM --bev-height-m = BEV camera altitude; higher = zooms OUT (reveals map-edge
REM void); lower = zooms IN so the map fills the panel. 600 fills the width
REM (a little of the taller map's top/bottom is cropped -- unavoidable on a
REM landscape panel). --bev-fov-deg 60 matches the square render's marker math.
"%VENV%\Scripts\interactive-drive.exe" --manifest "%MANIFEST%" --offload-text-encoder --bev-tilt-deg 0 --bev-height-m 1200 --bev-fov-deg 60 --game-mode %*
set EXIT_CODE=%ERRORLEVEL%

if not %EXIT_CODE%==0 ( echo. & echo interactive-drive exited with code %EXIT_CODE% & exit /b %EXIT_CODE% )
endlocal
