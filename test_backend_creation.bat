@echo off
setlocal enableextensions enabledelayedexpansion

cd /d C:\workspace\world\flashdream_public

set "VENV=C:\workspace\world\flashdream_public\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "PATH=%VENV%\Scripts;%PATH%"

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"

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

echo [TEST] Environment setup complete
"%PYEXE%" test_backend_creation.py
endlocal
