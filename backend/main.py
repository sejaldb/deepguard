from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from .model import predict  # relative import

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Deepfake API running"}

@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict(image)
    return result
