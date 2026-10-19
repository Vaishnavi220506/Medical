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
  pip install -q -r requirements.txt
fi
# NVIDIA GPU but PyTorch cannot use it: install the CUDA 12.8 build (new cards), else CUDA 12.6.
if command -v nvidia-smi >/dev/null 2>&1 && ! python scripts/check_gpu.py; then
  echo "Installing the GPU version of PyTorch for your NVIDIA card (about 3 GB, only once)..."
  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python scripts/check_gpu.py || pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi
python -c "import torch; print('AI models will run on:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else ('Apple GPU' if torch.backends.mps.is_available() else 'CPU'))"
[ -f .env ] || cp .env.example .env
[ -d demo_data ] || python scripts/make_demo_data.py
echo "Starting OncoVision at http://localhost:8000"
python -m uvicorn oncovision.api.main:app --host 127.0.0.1 --port 8000
