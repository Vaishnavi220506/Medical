#!/usr/bin/env bash
# One-command setup and start for macOS / Linux.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  if command -v nvidia-smi >/dev/null 2>&1; then
    pip install torch torchvision
  else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
[ -d demo_data ] || python scripts/make_demo_data.py
echo "Starting OncoVision at http://localhost:8000"
python -m uvicorn oncovision.api.main:app --host 127.0.0.1 --port 8000
