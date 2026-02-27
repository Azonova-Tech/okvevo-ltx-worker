FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

# Remove preinstalled conflicting packages first
RUN pip uninstall -y diffusers transformers accelerate || true

# Install everything from requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
