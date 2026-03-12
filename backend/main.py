from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from .model import predict_image, predict_from_url

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Deepfake API running"}

@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return predict_image(image)

@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    # Convert video to first frame for analysis
    import cv2
    contents = await file.read()
    with open("/tmp/temp_video.mp4", "wb") as f:
        f.write(contents)
    cap = cv2.VideoCapture("/tmp/temp_video.mp4")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"result": "error", "confidence": 0.0, "details": "Could not read video frame"}
    # Convert frame to PIL Image
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return predict_image(frame_image)

@app.post("/api/detect-url")
async def detect_url(url: str):
    return predict_from_url(url)

@app.post("/api/detect-camera-json")
async def detect_webcam_frame(frame: str):
    # frame is base64 string
    import base64
    image_data = base64.b64decode(frame)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return predict_image(image)

@app.post("/api/detect-screen-json")
async def detect_screen_frame(frame: str):
    # frame is base64 string
    import base64
    image_data = base64.b64decode(frame)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return predict_image(image)
