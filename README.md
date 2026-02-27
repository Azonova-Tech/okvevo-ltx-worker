# LTX-Video 0.9.8 RunPod Serverless Worker

RunPod serverless worker that generates videos using LTX-Video-0.9.8-2B-distilled (via `LTXConditionPipeline`) and uploads them to Firebase Storage.

---

## Prerequisites

- Docker + BuildKit enabled (`export DOCKER_BUILDKIT=1`)
- A Docker registry account (Docker Hub, GitHub Container Registry, etc.)
- A RunPod account with a serverless endpoint
- A HuggingFace account with access to [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video)
- A Firebase project with a service account and Storage bucket

---

## 1. Build the Docker Image

The model weights (~6.3 GB) are baked into the image at build time to eliminate cold-start download latency.

```bash
export DOCKER_BUILDKIT=1

docker build \
  --build-arg HF_TOKEN=your_huggingface_token \
  -t your-registry/ltx-worker:latest .
```

> **Alternative — Docker secrets (more secure):**
> ```bash
> echo "your_huggingface_token" > /tmp/hf_token
> docker build \
>   --secret id=hf_token,src=/tmp/hf_token \
>   -t your-registry/ltx-worker:latest .
> ```

---

## 2. Push to Registry

```bash
docker push your-registry/ltx-worker:latest
```

---

## 3. Create a RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Fill in:
   - **Container Image**: `your-registry/ltx-worker:latest`
   - **GPU**: `NVIDIA A40` or `RTX 4090` (24 GB VRAM minimum for 2B bfloat16)
   - **Container Disk**: `30 GB` (image is ~20 GB with model)
   - **Max Workers**: set based on your concurrency needs
   - **Idle Timeout**: `60` seconds (reduce cold starts)
   - **Execution Timeout**: `300` seconds (5 min per job)

4. Under **Environment Variables**, add:

| Variable | Value |
|---|---|
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Full JSON string of your Firebase service account key |
| `FIREBASE_STORAGE_BUCKET` | e.g. `your-project.appspot.com` |
| `HUGGINGFACE_HUB_TOKEN` | Your HuggingFace token (needed only if model is not baked in) |

5. Click **Deploy**

---

## 4. Call the Endpoint

**REST API:**
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A drone shot flying over a golden wheat field at sunset",
      "image_url": "https://example.com/frame.png",
      "job_id": "job_abc123",
      "scene_index": 0,
      "total_scenes": 3,
      "fps": 24,
      "output_path": "videos/job_abc123/scene_0.mp4"
    }
  }'
```

**Poll for result:**
```bash
curl https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
```

---

## Input Schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | Text prompt for video generation |
| `image_url` | string | ✅ | — | URL of the conditioning image (first frame) |
| `job_id` | string | ✅ | — | Firestore document ID for status tracking |
| `scene_index` | int | ✅ | — | Index of this scene within the job |
| `total_scenes` | int | ✅ | — | Total number of scenes in the job |
| `output_path` | string | ✅ | — | Firebase Storage path for the output video |
| `fps` | int | ❌ | `24` | Output video frame rate |

---

## Output Schema

**Success:**
```json
{
  "status": "success",
  "scene_index": 0,
  "video_url": "https://storage.googleapis.com/..."
}
```

**Error:**
```json
{
  "status": "error",
  "message": "..."
}
```

---

## Generation Defaults

| Setting | Value |
|---|---|
| Width | 832 |
| Height | 480 |
| Frames | 97 (8n+1) |
| Inference steps | 30 |
| Guidance scale | 1.0 (distilled) |
| Decode timestep | 0.05 |
| Image cond noise | 0.025 |
| dtype | bfloat16 |

---

## Network Volume (Optional — Skip Model Bake)

If you prefer to store model weights on a RunPod Network Volume instead of baking them into the image:

1. Create a Network Volume and mount it at `/runpod-volume`
2. Pre-download the model once:
   ```bash
   huggingface-cli download Lightricks/LTX-Video \
     ltxv-2b-0.9.8-distilled.safetensors \
     --local-dir /runpod-volume/models
   ```
3. Set the env var `MODEL_CACHE_DIR=/runpod-volume/models` on the endpoint
4. Build the image **without** the `HF_TOKEN` build arg (skip the bake step) — the worker will load from the volume at startup
