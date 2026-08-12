@echo off
setlocal enableextensions enabledelayedexpansion

cd /d C:\workspace\world\flashdream_public

set "VENV=C:\workspace\world\flashdream_public\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: flashdream .venv not found at %VENV% & exit /b 1 )

set "PATH=%VENV%\Scripts;%PATH%"

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0a"

set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;%LIB%"

set "PATH=C:\Users\kschmid\AppData\Local\ludus-renderer\physx-5.9.0\build-windows-AMD64\physx-lib\bin\win.x86_64.vc143.md\release;%PATH%"
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\x64\Microsoft.VC143.CRT;%PATH%"

set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if "%HF_TOKEN%"=="" if exist "C:\Users\kschmid\.cache\omni-dreams\huggingface\token" set /p HF_TOKEN=<"C:\Users\kschmid\.cache\omni-dreams\huggingface\token"

set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN"
set "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN"
set "TORCHINDUCTOR_MAX_AUTOTUNE=0"
set "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0"
set "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
set "TORCHINDUCTOR_CACHE_DIR=%~dp0.cache\torchinductor"
set "TRITON_CACHE_DIR=%~dp0.cache\triton"
set "TORCHINDUCTOR_COMPILE_THREADS=1"
if not exist "%~dp0.cache" mkdir "%~dp0.cache"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

set "MANIFEST=C:\workspace\world\flashdream_public\integrations\omnidreams\omnidreams\interactive_drive\configs\example_world_model_perf.yaml"

echo.
echo ===================================================================
echo PRECOMPILING TORCH.COMPILE CACHE (perf manifest)
echo ===================================================================
echo Manifest: %MANIFEST%
echo Cache dir: %~dp0.cache
echo This will take 2-3 minutes on first run, then warmup caches persist
echo ===================================================================
echo.

REM Run a single inference to trigger torch.compile and populate caches
"%PYEXE%" precompile_warmup.py

if %ERRORLEVEL% neq 0 (
  echo.
  echo [ERROR] Precompile failed with exit code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo.
echo ===================================================================
echo ✓ PRECOMPILE DONE - torch.compile cache is now warmed
echo Run run_interactive_drive_perf.bat for fast first chunk
echo ===================================================================
echo.

endlocal
