@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ===================================================================
echo Running test_load_state_dict.py on WSL2 Ubuntu
echo ===================================================================
echo.

cd /d C:\workspace\world\flashdream_public

wsl -e bash -c "sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip python3-venv >/dev/null 2>&1 ; cd /mnt/c/workspace/world/flashdream_public && python3 test_load_state_dict.py"

endlocal
