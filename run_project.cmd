@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0ai-service\run_server.cmd" (
  echo [ERROR] ai-service\run_server.cmd not found.
  pause
  exit /b 1
)

if not exist "%~dp0sillage-style\package.json" (
  echo [ERROR] sillage-style\package.json not found.
  pause
  exit /b 1
)

echo Starting Sillage^&Style...
echo.

start "SillageStyle Backend" cmd /k "cd /d "%~dp0ai-service" && call run_server.cmd"
timeout /t 2 /nobreak >nul
start "SillageStyle Frontend" cmd /k "cd /d "%~dp0sillage-style" && npm run dev"

echo.
echo Backend and frontend were started in separate windows.
echo Backend:  http://127.0.0.1:8000
echo Frontend: usually http://localhost:5173
echo.
echo Close this window when ready.
