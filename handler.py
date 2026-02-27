import os
import runpod
import torch
import json
import imageio
import firebase_admin
from firebase_admin import credentials, storage, firestore
from diffusers import LTXPipeline

# Firebase init
firebase_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(cred, {
    'storageBucket': os.environ["FIREBASE_STORAGE_BUCKET"]
})
bucket = storage.bucket()
db = firestore.client()

# Load model once
print("Loading LTX-Video model...")
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("Model loaded.")

def handler(event):
    inp = event["input"]
    prompt = inp["prompt"]
    job_id = inp["job_id"]          # e.g. "director-1234567"
    output_path = inp["output_path"] # e.g. "DirectorVideos/scene1-shot1.mp4"
    scene_index = inp["scene_index"] # 0, 1, 2...
    total_scenes = inp["total_scenes"]

    try:
        # Generate
        frames = pipe(
            prompt=prompt,
            num_frames=120,
            guidance_scale=3.0,
        ).frames[0]

        local_path = f"/tmp/{job_id}-scene{scene_index}.mp4"
        imageio.mimsave(local_path, frames, fps=24)

        # Upload to Firebase Storage at exact output_path
        blob = bucket.blob(output_path)
        blob.upload_from_filename(local_path, content_type="video/mp4")
        blob.make_public()
        video_url = blob.public_url

        # Update Firestore — Lambda watches this to trigger stitching
        db.collection("directorJobs").document(job_id).set({
            f"videoSlots": firestore.ArrayUnion([{
                "index": scene_index,
                "url": video_url,
                "outputPath": output_path,
                "status": "complete"
            }]),
            "totalScenes": total_scenes,
        }, merge=True)

        return {"status": "success", "video_url": video_url}

    except Exception as e:
        db.collection("directorJobs").document(job_id).set({
            f"videoSlots": firestore.ArrayUnion([{
                "index": scene_index,
                "status": "error",
                "error": str(e)
            }])
        }, merge=True)
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
