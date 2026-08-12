@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Precompile / warm the perf cache for run_interactive_drive_perf.bat.
REM Runs the PERF config HEADLESS (--stream-mjpeg, no Vulkan window) for a few
REM chunks so torch.compile's inductor kernels get built + written to the
REM PERSISTENT cache at C:\workspace\world\flashdream_public\.cache\torchinductor
REM (and .cache\triton). Then exits. The next real launch of
REM   C:\workspace\world\flashdream_public\run_interactive_drive_perf.bat
REM reuses those compiled kernels and skips the ~minute compile warmup.
REM
REM Usage:
REM   C:\workspace\world\flashdream_public\run_interactive_drive_perf_precompile.bat
REM   C:\workspace\world\flashdream_public\run_interactive_drive_perf_precompile.bat 5   (warm N chunks)
REM ==========================================================================
set "CHUNKS=%~1"
if "%CHUNKS%"=="" set "CHUNKS=3"
echo Warming the perf compile cache for %CHUNKS% chunks (headless, no window)...
REM --stream-mjpeg on a throwaway port = headless (no Vulkan); --stop-after-chunks
REM exits cleanly once N chunks are generated (chunk 0 is the warmup chunk).
REM --auto-start drives the default scene immediately (headless has no browser to
REM pick one, so without this it just idles at "waiting for first scene selection"
REM and never compiles). It generates chunks -> compiles the DiT kernels -> stops.
call "%~dp0run_interactive_drive_perf.bat" --auto-start --stream-mjpeg 127.0.0.1:8799 --stop-after-chunks %CHUNKS% --no-hud --game-mode
echo.
echo Cache warmed. Now launch normally (fast start):
echo   C:\workspace\world\flashdream_public\run_interactive_drive_perf.bat
endlocal
