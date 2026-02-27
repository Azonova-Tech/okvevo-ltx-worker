FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# HuggingFace cache lives inside the image layer
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .

# Remove preinstalled conflicting packages first
RUN pip uninstall -y diffusers transformers accelerate || true

# Install everything from requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY download_model.py .

# Pre-bake model weights into the image (requires --build-arg HF_TOKEN=...)
ARG HF_TOKEN
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN_SECRET=$(cat /run/secrets/hf_token 2>/dev/null || echo "${HF_TOKEN}") && \
    HUGGINGFACE_HUB_TOKEN=${HF_TOKEN_SECRET} python download_model.py

CMD ["python", "-u", "handler.py"]
