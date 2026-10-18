FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU build of PyTorch keeps the image small; use a CUDA base image for GPU servers.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY oncovision ./oncovision
COPY frontend ./frontend
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "oncovision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
