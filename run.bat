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
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || goto :error
  ) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :error
  )
  pip install -r requirements.txt || goto :error
) else (
  call .venv\Scripts\activate.bat
)
if not exist .env copy .env.example .env >nul
if not exist demo_data python scripts\make_demo_data.py
echo Starting OncoVision at http://localhost:8000
python -m uvicorn oncovision.api.main:app --host 127.0.0.1 --port 8000
goto :eof
:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
