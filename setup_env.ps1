# ============================================================
# One-click env setup: Python 3.12 + CUDA torch + ultralytics
# Purpose: build GPU environment for YOLOv8 training (RTX 3050 / CUDA 12.x)
#
# Usage (run in YOUR OWN PowerShell terminal, not the sandbox):
#   cd d:\19701\Source\Repos\dam-particle-detection
#   .\setup_env.ps1
#
# Note: this file is intentionally ASCII-only to avoid Windows PowerShell 5.1
# GBK codepage mis-reading UTF-8 Chinese (which breaks script parsing).
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  YOLOv8 Training Env - One-click Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ---- 1. Locate Python 3.12 ----
$py312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py312)) {
    Write-Host "[X] Python 3.12 not found, installing via winget..." -ForegroundColor Yellow
    winget install Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    if (-not (Test-Path $py312)) {
        Write-Host "[X] Python 3.12 install failed, please install Python 3.12 manually" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK] Python 3.12: $py312" -ForegroundColor Green
& $py312 --version

# ---- 2. Create virtual environment ----
$venvDir = ".venv312"
$venvPython = "$venvDir\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "`n[>>] Creating virtual environment $venvDir ..." -ForegroundColor Yellow
    & $py312 -m venv $venvDir
} else {
    Write-Host "[OK] Virtual environment already exists: $venvDir" -ForegroundColor Green
}

# ---- 3. Upgrade pip ----
Write-Host "`n[>>] Upgrading pip ..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

# ---- 4. Install CUDA PyTorch (cu124) ----
Write-Host "`n[>>] Installing PyTorch CUDA 12.4 (~2.5GB, please be patient)..." -ForegroundColor Yellow
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --timeout 600 --retries 10

# ---- 5. Install ultralytics and project deps ----
Write-Host "`n[>>] Installing ultralytics and project deps ..." -ForegroundColor Yellow
& $venvPython -m pip install ultralytics opencv-python matplotlib scipy scikit-image pandas pyyaml tqdm

# ---- 6. Verify CUDA ----
Write-Host "`n[>>] Verifying CUDA availability ..." -ForegroundColor Yellow
$pycode = @'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
'@
& $venvPython -c $pycode

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Environment setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext step: run training" -ForegroundColor Cyan
Write-Host "  Quick (yolov8n):  .\.venv312\Scripts\python.exe train_yolo.py --stage quick" -ForegroundColor White
Write-Host "  Final (yolov8x):  .\.venv312\Scripts\python.exe train_yolo.py --stage final" -ForegroundColor White
