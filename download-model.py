import os
from huggingface_hub import hf_hub_download

MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/hf_cache/models")
FILENAME = "ltxv-2b-0.9.8-distilled.safetensors"
TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

dest = os.path.join(MODEL_CACHE_DIR, FILENAME)
if os.path.exists(dest):
    print(f"Model already present at {dest}, skipping download.")
else:
    print(f"Downloading {FILENAME} from Lightricks/LTX-Video ...")
    hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=FILENAME,
        local_dir=MODEL_CACHE_DIR,
        token=TOKEN or None,
    )
    print(f"Saved to {dest}")
