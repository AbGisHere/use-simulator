# NSE Simulator Launcher - Windows PowerShell
# Usage: Right-click -> "Run with PowerShell"  OR  run from terminal:
#   powershell -ExecutionPolicy Bypass -File start.ps1
#
# Requirements: Python 3.11+, Node.js 18+

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BackendDir  = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"

Write-Host ""
Write-Host "+======================================+" -ForegroundColor Green
Write-Host "|        NSE Simulator Launcher        |" -ForegroundColor Green
Write-Host "+======================================+" -ForegroundColor Green
Write-Host ""

# -- Check prerequisites -------------------------------------------------------

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js not found. Install Node.js 18+ from nodejs.org" -ForegroundColor Red
    exit 1
}

# -- Backend setup -------------------------------------------------------------

Write-Host "Setting up backend..." -ForegroundColor Yellow
Set-Location $BackendDir

# Create .env if missing
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created backend\.env from .env.example - edit it to add API keys." -ForegroundColor Yellow
}

# Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    } else {
        python -m venv .venv
    }
}

# Install/upgrade dependencies
Write-Host "Installing Python dependencies..."
& ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
& ".venv\Scripts\pip.exe" install -q -r requirements.txt

Write-Host "Backend ready." -ForegroundColor Green

# -- Frontend setup -------------------------------------------------------------

Write-Host "Setting up frontend..." -ForegroundColor Yellow
Set-Location $FrontendDir

if (-not (Test-Path "node_modules\.bin\next")) {
    Write-Host "Installing Node.js dependencies..."
    npm install --legacy-peer-deps
}

Write-Host "Frontend ready." -ForegroundColor Green

# -- Launch both services ------------------------------------------------------

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: First stock addition downloads the FinBERT model (~500MB)." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both services."
Write-Host ""

# Start backend in this window
$backendProc = Start-Process "$BackendDir\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory $BackendDir -NoNewWindow -PassThru

# Brief pause for backend to start
Start-Sleep -Seconds 3

# Start frontend in this window
$frontendProc = Start-Process "cmd.exe" -ArgumentList "/c npm run dev" -WorkingDirectory $FrontendDir -NoNewWindow -PassThru

Write-Host "Both services are running in this window." -ForegroundColor Green
Write-Host "Press any key to stop both services..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup
Write-Host "Shutting down..."
Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue
