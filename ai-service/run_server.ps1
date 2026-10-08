# Sillage&Style local AI service launcher.
# Uses a separate .venv_ai environment and a PyTorch-compatible Python version.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$VenvDir = Join-Path $Root ".venv_ai"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$Supported = @("3.12", "3.11", "3.10", "3.13")

function Find-CompatiblePython {
    foreach ($Version in $Supported) {
        & py "-$Version" -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $Version
        }
    }
    return $null
}

$Selected = Find-CompatiblePython
if (-not $Selected) {
    Write-Host ""
    Write-Host "Compatible Python was not found." -ForegroundColor Red
    Write-Host "This project currently requires Python 3.10-3.13 for torch<2.7."
    Write-Host "Recommended version: Python 3.12."
    Write-Host ""
    Write-Host "Install Python 3.12, then run this script again."
    Write-Host "After installation you can verify it with: py -0p"
    exit 1
}

if (Test-Path $Py) {
    $VenvVersion = (& $Py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($Supported -notcontains $VenvVersion) {
        Write-Host "Existing .venv_ai uses unsupported Python $VenvVersion. Recreating it..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path $Py)) {
    Write-Host "Creating .venv_ai with Python $Selected ..."
    & py "-$Selected" -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Installing backend dependencies..."
& $Py -m pip install --upgrade pip -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Py -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Swagger: http://127.0.0.1:8000/docs"
Write-Host "Stop server: Ctrl+C"
Write-Host ""
Write-Host "Run frontend in another terminal: cd ..\sillage-style ; npm run dev"

& $Py -m uvicorn app:app --host 127.0.0.1 --port 8000
