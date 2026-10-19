@echo off
REM One-command setup and start for Windows.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :error
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  where nvidia-smi >nul 2>nul
  if %errorlevel%==0 (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :error
  ) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :error
  )
  pip install -r requirements.txt || goto :error
) else (
  call .venv\Scripts\activate.bat
  python -m pip install -q -r requirements.txt
)
REM NVIDIA GPU but PyTorch cannot use it (CPU-only build, or a build without kernels for a new
REM card such as the RTX 50 series): install the CUDA 12.8 build, or CUDA 12.6 for older cards.
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  python scripts\check_gpu.py
  if errorlevel 1 (
    echo Installing the GPU version of PyTorch for your NVIDIA card, about 3 GB, only once...
    pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    python scripts\check_gpu.py
    if errorlevel 1 (
      echo Trying the PyTorch build for older NVIDIA cards...
      pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
      python scripts\check_gpu.py
    )
  )
)
python -c "import torch; print('AI models will run on:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if not exist .env copy .env.example .env >nul
if not exist demo_data python scripts\make_demo_data.py
echo Starting OncoVision at http://localhost:8000
python -m uvicorn oncovision.api.main:app --host 127.0.0.1 --port 8000
goto :eof
:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
