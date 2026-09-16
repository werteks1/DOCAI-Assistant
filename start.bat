@echo off
chcp 65001 >nul
cd /d "%~dp0"
title DocAI Assistant

if exist "venv\Scripts\python.exe" goto deps

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 goto fail_venv
"venv\Scripts\python.exe" -m pip install --upgrade pip

:deps
echo Installing dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail_deps

if exist "web\dist\index.html" goto run

echo Building web interface...
pushd web
call npm install
if errorlevel 1 goto fail_npm
call npm run build
if errorlevel 1 goto fail_build
popd

:run
echo Starting DocAI server: http://127.0.0.1:8000
echo First login: admin / admin
echo To stop the server close this window or press Ctrl+C
echo.
start "DocAI" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"venv\Scripts\python.exe" -m compiler
pause
exit /b 0

:fail_venv
echo Failed to create venv. Install Python 3.12 and tick "Add python.exe to PATH".
pause
exit /b 1

:fail_deps
echo Failed to install dependencies. Check the internet connection.
pause
exit /b 1

:fail_npm
popd
echo Failed to install npm packages. Install Node.js LTS from https://nodejs.org
pause
exit /b 1

:fail_build
popd
echo Failed to build the web interface.
pause
exit /b 1
