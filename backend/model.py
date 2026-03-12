import os
import requests
from PIL import Image
import base64
from io import BytesIO

HF_TOKEN = os.getenv("HF_API_TOKEN")  # optional if model is private
MODEL = "dima806/deepfake_vs_real_image_detection"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def predict(image: Image.Image):
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Call Hugging Face Inference API
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": img_str}
    )

    try:
        data = response.json()
        # The API returns a list of predictions
        top = data[0] if isinstance(data, list) else {"label": "error", "score": 0.0}
        return {"result": top.get("label", "error"), "confidence": float(top.get("score", 0.0))}
    except Exception as e:
        return {"result": "error", "confidence": 0.0, "details": str(e)}

