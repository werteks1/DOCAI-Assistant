@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title DocAI Assistant

echo ============================================
echo   DocAI Assistant - запуск на Windows
echo ============================================
echo.

rem --- Python 3.10+ -------------------------------------------------
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 goto no_python
goto python_ok

:no_python
echo [Ошибка] Python 3.10 или новее не найден.
echo Установите его с https://www.python.org/downloads/windows/
echo и при установке отметьте галочку "Add python.exe to PATH".
echo.
pause
exit /b 1

:python_ok

rem --- Виртуальное окружение и зависимости ---------------------------
if not exist "venv\Scripts\python.exe" (
  echo Создаю виртуальное окружение venv...
  %PY% -m venv venv
  if errorlevel 1 (
    echo [Ошибка] Не удалось создать venv. Проверьте установку Python.
    pause
    exit /b 1
  )
  echo Устанавливаю зависимости (при первом запуске это занимает несколько минут)...
  "venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  "venv\Scripts\python.exe" -m pip install -r requirements-web.txt
  if errorlevel 1 (
    echo [Ошибка] Не удалось установить зависимости. Проверьте интернет-соединение.
    pause
    exit /b 1
  )
)

rem --- Собранный интерфейс (web\dist) --------------------------------
if not exist "web\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [Внимание] Node.js не найден, а собранный интерфейс отсутствует.
    echo Установите Node.js LTS с https://nodejs.org и запустите файл снова.
    pause
    exit /b 1
  )
  echo Собираю интерфейс (npm build)...
  pushd web
  if exist package-lock.json (call npm ci) else (call npm install)
  if errorlevel 1 (
    popd
    echo [Ошибка] Не удалось установить npm-зависимости.
    pause
    exit /b 1
  )
  call npm run build
  if errorlevel 1 (
    popd
    echo [Ошибка] Не удалось собрать интерфейс.
    pause
    exit /b 1
  )
  popd
)

rem --- Запуск ---------------------------------------------------------
echo.
echo Сервер запускается. Интерфейс: http://127.0.0.1:8000
echo Вход в первый раз: admin / admin (потребуется сменить пароль).
echo Если Windows спросит про брандмауэр - разрешите доступ (нужен для класса).
echo Для остановки закройте это окно или нажмите Ctrl+C.
echo.

start "DocAI браузер" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"venv\Scripts\python.exe" -m compiler

echo.
echo Сервер остановлен.
pause
