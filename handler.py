import os


print("Starting worker...")

print("Checking environment variables...")

required_vars = [
    "FIREBASE_SERVICE_ACCOUNT_JSON",
    "FIREBASE_STORAGE_BUCKET",
    "HUGGINGFACE_HUB_TOKEN"
]

for var in required_vars:
    if var not in os.environ:
        raise Exception(f"Missing environment variable: {var}")

print("Environment variables OK.")

import runpod
import torch
import json
import uuid
import imageio
import firebase_admin
from firebase_admin import credentials, storage, firestore
from diffusers import LTXPipeline

# -----------------------------
# Firebase Init
# -----------------------------
firebase_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
cred = credentials.Certificate(firebase_dict)

firebase_admin.initialize_app(cred, {
    "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"]
})

bucket = storage.bucket()
db = firestore.client()

# -----------------------------
# Load Model Once (Cold Start)
# -----------------------------
print("Loading LTX-Video model...")

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

print("Model loaded successfully.")

# -----------------------------
# Video Generation
# -----------------------------
def generate_scene(prompt, frames=120):
    output = pipe(
        prompt=prompt,
        num_frames=frames,
        guidance_scale=3.0,
    )
    return output.frames[0]

# -----------------------------
# RunPod Handler
# -----------------------------
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

        # Generate 5s chunk (120 frames @ 24fps)
        frames = generate_scene(prompt, frames=120)

        file_name = f"{job_id}-scene-{scene_index}.mp4"
        local_path = f"/tmp/{file_name}"

        imageio.mimsave(local_path, frames, fps=fps)

        # Upload to Firebase Storage
        blob = bucket.blob(output_path)
        blob.upload_from_filename(local_path, content_type="video/mp4")
        blob.make_public()

        video_url = blob.public_url

        # Update Firestore
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
