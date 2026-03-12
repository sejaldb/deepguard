# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import tempfile
from PIL import Image

from model import predict
from utils import download_image, extract_frames_from_video

app = FastAPI(title="Deepfake Detection API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Test endpoint
# --------------------------
@app.get("/")
def home():
    return {"message": "Deepfake API running"}

# --------------------------
# Detect Image
# --------------------------
@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return predict(image)

# --------------------------
# Detect Video
# --------------------------
@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    frames = extract_frames_from_video(temp_path)
    results = [predict(frame) for frame in frames]

    # Aggregate results
    fake_count = sum(1 for r in results if r['result'] == 'FAKE')
    real_count = sum(1 for r in results if r['result'] == 'REAL')
    overall = "FAKE" if fake_count > real_count else "REAL"
    confidence = max(fake_count, real_count) / len(results) if results else 0

    return {"result": overall, "confidence": confidence, "frames_analyzed": len(results)}

# --------------------------
# Detect from URL
# --------------------------
@app.post("/api/detect-url")
async def detect_url(url_data: dict):
    url = url_data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL missing")
    try:
        image = download_image(url)
    except:
        raise HTTPException(status_code=400, detail="Cannot download image from URL")
    return predict(image)

# --------------------------
# Detect Webcam / Screen Frames (base64)
# --------------------------
@app.post("/api/detect-camera-json")
async def detect_camera(frame: dict):
    frame_b64 = frame.get("frame")
    if not frame_b64:
        raise HTTPException(status_code=400, detail="Frame missing")
    img_bytes = base64.b64decode(frame_b64)
    image = Image.open(io.BytesIO(img_bytes))
    return predict(image)

@app.post("/api/detect-screen-json")
async def detect_screen(frame: dict):
    frame_b64 = frame.get("frame")
    if not frame_b64:
        raise HTTPException(status_code=400, detail="Frame missing")
    img_bytes = base64.b64decode(frame_b64)
    image = Image.open(io.BytesIO(img_bytes))
    return predict(image)