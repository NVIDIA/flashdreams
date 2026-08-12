@echo off
setlocal enableextensions enabledelayedexpansion

cd /d C:\workspace\world\flashdream_public

set "VENV=C:\workspace\world\flashdream_public\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: flashdream .venv not found at %VENV% & exit /b 1 )

set "PATH=%VENV%\Scripts;%PATH%"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

echo.
echo ===================================================================
echo DOWNLOAD ALL HUGGINGFACE MODELS FOR FLASHDREAM
echo ===================================================================
echo.
echo This will download ~50-100 GB of models (takes 1-2 hours)
echo Cache location: %USERPROFILE%\.cache\huggingface
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

"%PYEXE%" download_all_models.py
if %ERRORLEVEL% neq 0 ( echo. & echo Download failed with exit code %ERRORLEVEL% & exit /b %ERRORLEVEL% )

echo.
echo ===================================================================
echo MODELS DOWNLOADED - Now run setup.bat
echo ===================================================================
echo.
endlocal
