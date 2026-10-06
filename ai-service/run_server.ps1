# Запуск AI-сервиса без зависимости от повреждённой папки .venv.
# Использует отдельное окружение .venv_ai (рядом с app.py).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$VenvDir = Join-Path $Root ".venv_ai"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "Создаю виртуальное окружение .venv_ai ..."
    py -m venv $VenvDir
}

& $Py -m pip install --upgrade pip -q
& $Py -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host "Сервер: http://127.0.0.1:8000  (останов: Ctrl+C)"
Write-Host "Сайт запускайте отдельно: cd ..\sillage-style  ->  npm run dev"
& $Py -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
