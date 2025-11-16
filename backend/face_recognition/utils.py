import io
from PIL import Image
import numpy as np
import face_recognition

def read_imagefile_bytes(upload_file):
    """Read uploaded file → convert to RGB numpy array"""
    content = upload_file.file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)

def get_face_embeddings(img_np):
    """
    Detect faces and return embeddings (128D vectors)
    """
    face_locations = face_recognition.face_locations(img_np, model="hog")
    if not face_locations:
        return []
    encodings = face_recognition.face_encodings(img_np, known_face_locations=face_locations)
    return encodings
