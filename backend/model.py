import os
from transformers import pipeline
from PIL import Image

# Load token from environment variable (optional if private model)
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize classifier
classifier = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
    use_auth_token=HF_TOKEN  # set None if model is public
)

def predict(image: Image.Image):
    image = image.convert("RGB")
    results = classifier(image)
    top = results[0]
    return {"result": top["label"], "confidence": float(top["score"])}
