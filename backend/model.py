import os
import requests
from PIL import Image
import base64
from io import BytesIO

HF_TOKEN = os.getenv("HF_API_TOKEN")  # optional if model is private
MODEL = "dima806/deepfake_vs_real_image_detection"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def predict_image(image: Image.Image):
    """
    Predict a single image (PIL.Image)
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": img_str})

    try:
        data = response.json()
        top = data[0] if isinstance(data, list) else {"label": "error", "score": 0.0}
        return {"result": top.get("label", "error"), "confidence": float(top.get("score", 0.0))}
    except Exception as e:
        return {"result": "error", "confidence": 0.0, "details": str(e)}

def predict_from_url(url: str):
    """
    Predict from an image/video URL
    """
    try:
        import requests
        from PIL import Image
        from io import BytesIO

        resp = requests.get(url)
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return predict_image(image)
    except Exception as e:
        return {"result": "error", "confidence": 0.0, "details": str(e)}
