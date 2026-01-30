import threading
import numpy as np
import cv2
from insightface.app import FaceAnalysis

from .config import INSIGHTFACE_MODEL

_analyzer = None
_lock = threading.Lock()

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def get_analyzer() -> FaceAnalysis:
    global _analyzer
    if _analyzer is None:
        with _lock:
            if _analyzer is None:
                # CPU-only
                _analyzer = FaceAnalysis(
                    name=INSIGHTFACE_MODEL,
                    providers=["CPUExecutionProvider"],
                )
                _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer

def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image (could not decode)")
    return img

def extract_single_embedding(img_bgr: np.ndarray) -> np.ndarray:
    app = get_analyzer()
    faces = app.get(img_bgr)

    if len(faces) == 0:
        raise ValueError("No face detected")
    if len(faces) > 1:
        raise ValueError("Multiple faces detected (only 1 face allowed)")

    emb = faces[0].embedding.astype(np.float32)
    return _l2_normalize(emb)
