@echo off
REM GraphRiverCast — Setup script for Windows
REM Usage: scripts\setup_windows.bat

echo ========================================================
echo        GraphRiverCast Environment Setup (Windows)
echo ========================================================
echo.

REM ── Check conda ──
where conda >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [1/4] Conda found
    echo       Creating environment from environment.yml ...
    conda env create -f environment.yml || conda env update -f environment.yml
    echo.
    echo       Activate with: conda activate graphrivercast
) else (
    echo [1/4] Conda not found, using pip ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

REM ── Check CUDA ──
echo.
echo [2/4] Checking CUDA ...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
) else (
    echo       No NVIDIA GPU detected (CPU-only mode available)
)

REM ── Verify ──
echo.
echo [3/4] Verifying installation ...
python -c "import torch; import torch_geometric; from src.model import GraphRiverCast; print('OK')" 2>nul
if %ERRORLEVEL% equ 0 (
    echo       All imports successful
) else (
    echo       Verification skipped (activate environment first)
)

echo.
echo [4/4] Setup complete!
echo.
echo ========================================================
echo   Quick start:
echo     python -m src.inference ^
echo       --checkpoint checkpoints\pretrain\GRC_ColdStart.ckpt ^
echo       --data-dir .\data\global ^
echo       --start-date 2015-01-01
echo ========================================================
pause
