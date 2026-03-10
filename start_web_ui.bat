@echo off
setlocal

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_EXE=%ROOT_DIR%..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python virtual environment not found.
  echo Expected one of:
  echo   %ROOT_DIR%..\.venv\Scripts\python.exe
  echo   %ROOT_DIR%.venv\Scripts\python.exe
  pause
  exit /b 1
)

set "APP_PATH=%ROOT_DIR%cpt_unimol_project\web_ui\app.py"
if not exist "%APP_PATH%" (
  echo [ERROR] App file not found: %APP_PATH%
  pause
  exit /b 1
)

set "PORT=7860"
set "HAS_SERVER="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  set "HAS_SERVER=1"
)

if not defined HAS_SERVER (
  start "CPT Web UI" "%PYTHON_EXE%" "%APP_PATH%"
  timeout /t 3 /nobreak >nul
)

start "" "http://127.0.0.1:%PORT%"
exit /b 0
