# backend/utils.py
import cv2
import requests
from PIL import Image
from io import BytesIO

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def extract_frames_from_video(video_path: str, frame_interval=30):
    """
    Extract frames from video every `frame_interval` frames
    Returns a list of PIL Images
    """
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
        count += 1

    vidcap.release()
    return frames