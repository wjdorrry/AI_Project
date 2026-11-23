@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0..\ml\.venv_week7\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Working Python environment was not found:
  echo %PY%
  echo.
  echo Expected environment: ml\.venv_week7
  pause
  exit /b 1
)

echo Checking backend dependencies...
"%PY%" -c "import fastapi, uvicorn, multipart, torch, transformers; print('Backend deps OK'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
if errorlevel 1 (
  echo.
  echo [ERROR] Backend dependencies are missing.
  echo Run:
  echo "%PY%" -m pip install fastapi "uvicorn[standard]" python-multipart
  pause
  exit /b 1
)

echo.
echo Starting Sillage^&Style backend...
echo Backend: http://127.0.0.1:8000
echo Swagger: http://127.0.0.1:8000/docs
echo Stop: Ctrl+C
echo.

"%PY%" -m uvicorn app:app --host 127.0.0.1 --port 8000
