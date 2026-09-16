@echo off
chcp 65001 >nul
cd /d "%~dp0"
title DocAI Assistant

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Install Python 3.12 with "Add python.exe to PATH" checked.
        pause
        exit /b 1
    )
    "venv\Scripts\python.exe" -m pip install --upgrade pip
)

echo Installing dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Check the internet connection and the Python version (3.12 recommended).
    pause
    exit /b 1
)

if not exist "web\dist\index.html" (
    echo Building web interface...
    pushd web
    call npm install
    if errorlevel 1 (
        popd
        echo Failed to install npm packages. Install Node.js LTS from https://nodejs.org
        pause
        exit /b 1
    )
    call npm run build
    if errorlevel 1 (
        popd
        echo Failed to build the web interface.
        pause
        exit /b 1
    )
    popd
)

start "DocAI" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"venv\Scripts\python.exe" -m compiler
pause
