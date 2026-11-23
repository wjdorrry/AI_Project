@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%ai-service\run_server.cmd" (
  echo [ERROR] ai-service\run_server.cmd not found.
  pause
  exit /b 1
)

if not exist "%ROOT%sillage-style\package.json" (
  echo [ERROR] sillage-style\package.json not found.
  pause
  exit /b 1
)

echo Starting Sillage^&Style...
echo.

start "SillageStyle Backend" /D "%ROOT%ai-service" cmd /k call run_server.cmd
timeout /t 3 /nobreak >nul
start "SillageStyle Frontend" /D "%ROOT%sillage-style" cmd /k npm run dev

echo.
echo Backend and frontend were started in separate windows.
echo Backend:  http://127.0.0.1:8000
echo Frontend: usually http://localhost:5173
echo.
echo If login is opened immediately, give the backend a few seconds to start.
echo.
