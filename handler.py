import os
import runpod
import torch
import json
import imageio
import firebase_admin
from firebase_admin import credentials, storage, firestore
from diffusers import LTXConditionPipeline

print("Starting worker...")

# -------------------------------------------------
# Check Required Environment Variables
# -------------------------------------------------
required_vars = [
    "FIREBASE_SERVICE_ACCOUNT_JSON",
    "FIREBASE_STORAGE_BUCKET",
    "HUGGINGFACE_HUB_TOKEN"
]

for var in required_vars:
    if var not in os.environ:
        raise Exception(f"Missing environment variable: {var}")

print("Environment variables OK.")

# -------------------------------------------------
# Firebase Init
# -------------------------------------------------
firebase_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
cred = credentials.Certificate(firebase_dict)

firebase_admin.initialize_app(cred, {
    "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"]
})

bucket = storage.bucket()
db = firestore.client()

# -------------------------------------------------
# Load LTX 0.9.8 Distilled (Stable Version)
# -------------------------------------------------
print("Loading LTX-Video-0.9.8-distilled model...")

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-distilled",
    torch_dtype=torch.float16,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

pipe.to("cuda")
pipe.vae.enable_tiling()

print("Model loaded successfully.")

# -------------------------------------------------
# Generation Settings (Safe Defaults)
# -------------------------------------------------
WIDTH = 832        # must be divisible by 32
HEIGHT = 480       # must be divisible by 32
NUM_FRAMES = 97    # must follow 8n + 1 rule
INFERENCE_STEPS = 30

# -------------------------------------------------
# RunPod Handler
# -------------------------------------------------
def handler(event):
    inp = event["input"]

    prompt = inp["prompt"]
    job_id = inp["job_id"]
    scene_index = inp["scene_index"]
    total_scenes = inp["total_scenes"]
    fps = inp.get("fps", 24)
    output_path = inp["output_path"]

    try:
        # Mark processing
        db.collection("directorJobs").document(job_id).set({
            f"scenes.{scene_index}.status": "processing"
        }, merge=True)

        print("Generating video...")

        video = pipe(
            prompt=prompt,
            negative_prompt="worst quality, blurry, distorted, jittery",
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            num_inference_steps=INFERENCE_STEPS,
        ).frames[0]

        print("Generation complete.")

        local_path = f"/tmp/{job_id}-scene-{scene_index}.mp4"
        imageio.mimsave(local_path, video, fps=fps)

        blob = bucket.blob(output_path)
        blob.upload_from_filename(local_path, content_type="video/mp4")
        blob.make_public()

        video_url = blob.public_url

        db.collection("directorJobs").document(job_id).set({
            f"scenes.{scene_index}": {
                "status": "complete",
                "url": video_url,
                "outputPath": output_path
            },
            "totalScenes": total_scenes
        }, merge=True)

        return {
            "status": "success",
            "scene_index": scene_index,
            "video_url": video_url
        }

    except Exception as e:
        print("Error:", str(e))

        db.collection("directorJobs").document(job_id).set({
            f"scenes.{scene_index}": {
                "status": "error",
                "error": str(e)
            }
        }, merge=True)

        return {
            "status": "error",
            "message": str(e)
        }

runpod.serverless.start({"handler": handler})
