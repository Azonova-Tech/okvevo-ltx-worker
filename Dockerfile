FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .

# Remove preinstalled conflicting packages first, then force-reinstall
RUN pip uninstall -y diffusers transformers accelerate || true

# Install everything from requirements (--force-reinstall ensures base image versions are overwritten)
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
