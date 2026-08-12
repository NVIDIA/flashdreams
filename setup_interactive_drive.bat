@echo off
setlocal enableextensions enabledelayedexpansion

cd /d C:\workspace\world\flashdream_public

set "VENV=C:\workspace\world\flashdream_public\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: flashdream .venv not found at %VENV% & exit /b 1 )

REM Setup CUDA and environment (same as run_interactive_drive_perf.bat)
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
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

echo.
echo ===================================================================
echo OMNIDREAMS INTERACTIVE-DRIVE SETUP
echo ===================================================================
echo.

REM Check HF_TOKEN
if "%HF_TOKEN%"=="" (
  if exist "C:\Users\kschmid\.cache\omni-dreams\huggingface\token" (
    set /p HF_TOKEN=<"C:\Users\kschmid\.cache\omni-dreams\huggingface\token"
    echo [SETUP] ✓ Loaded HF_TOKEN from cache
  ) else (
    echo [SETUP] ⚠ HF_TOKEN not set. Set it manually or the setup will fail:
    echo   set HF_TOKEN=your-token-here
    echo.
  )
)

REM Step 1: Sync dependencies (narrow sync preserves pinned torch version)
echo [SETUP] 1. Syncing dependencies...
uv sync --package flashdreams-omnidreams --extra dev --extra interactive-drive
if %ERRORLEVEL% neq 0 ( echo [ERROR] uv sync failed & exit /b %ERRORLEVEL% )

REM Step 1b: Install SageAttention (optimized attention backend for inference)
echo.
echo [SETUP] 1b. Installing SageAttention (optional, for faster inference)...
uv pip install sageattention --no-deps
if %ERRORLEVEL% neq 0 ( echo [WARN] SageAttention install failed, continuing without it )

REM Step 2: Sync third-party sources
echo.
echo [SETUP] 2. Syncing third-party sources...
uv run --package flashdreams-omnidreams python integrations/omnidreams/omnidreams_singleview/tools/sync_thirdparty.py sync
if %ERRORLEVEL% neq 0 ( echo [ERROR] sync_thirdparty failed & exit /b %ERRORLEVEL% )

REM Step 3: Prepare for perf
echo.
echo [SETUP] 3. Preparing for perf (downloads models, builds extensions)...
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
if %ERRORLEVEL% neq 0 ( echo [ERROR] omnidreams-prepare failed & exit /b %ERRORLEVEL% )

REM Step 4: Optional precompile torch.compile cache
echo.
echo [SETUP] 4. Precompiling torch.compile cache (optional)...
choice /C YN /M "Warmup torch.compile cache? (faster first chunk, takes 2-3 min) [Y/N]: "
if %ERRORLEVEL%==1 (
  call .\precompile_cache.bat
  if %ERRORLEVEL% neq 0 ( echo [WARN] Precompile failed, continuing anyway )
)

echo.
echo ===================================================================
echo ✓ SETUP COMPLETE
echo ===================================================================
echo.
echo Next: Run the interactive-drive app
echo   .\run_interactive_drive_perf.bat --game-mode
echo.
echo Controls: WASD=drive Mouse=look C=obstacle R=restart Esc=quit
echo Editing: Type in Scene Prompt field, /spawn car 30 5, /clear-actors
echo.
endlocal
