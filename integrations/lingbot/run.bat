@echo off
setlocal enabledelayedexpansion

set HERE=%~dp0
cd /d "%HERE%"

call .venv\Scripts\activate.bat

set PYTHONPATH=%HERE%;%HERE%..\..\apps;%PYTHONPATH%

python -m lingbot.runner
