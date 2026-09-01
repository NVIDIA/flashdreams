@echo off
setlocal enabledelayedexpansion

set HERE=%~dp0
cd /d "%HERE%"

echo.
echo ==========================================
echo LingBot WebRTC - Setup
echo ==========================================
echo.

echo [1/5] Cleaning old setup...
if exist .venv rmdir /s /q .venv >nul 2>&1

echo [2/5] Creating Python venv...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo [4/5] Installing PyTorch with CUDA 13.2...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132

echo [5/5] Installing dependencies...
pip install "transformers>=5.0,<6" sentencepiece scipy opencv-python
pip install aiohttp aiortc python-multipart loguru
pip install tyro pydantic fastapi uvicorn pillow numpy gradio websockets
pip install flash-attn==2.6.3 --no-build-isolation 2>nul || echo flash-attn skipped

echo.
echo ==========================================
echo ✓ SETUP COMPLETE
echo ==========================================
echo.
echo To run server: run.bat
echo.
