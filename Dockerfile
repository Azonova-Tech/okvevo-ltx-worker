FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV PIP_DEFAULT_TIMEOUT=1000

COPY requirements.txt .

# Remove preinstalled conflicting packages
RUN pip uninstall -y diffusers transformers accelerate || true

# Upgrade pip (important for large installs)
RUN pip install --upgrade pip

# Install dependencies with retries and longer timeout
RUN pip install --no-cache-dir --retries 10 --timeout 1000 --force-reinstall -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
