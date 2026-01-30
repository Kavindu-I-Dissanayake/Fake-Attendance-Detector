import os

CLASS_NAME = "Group-Project"

# Mongo
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fake_attendance_system")

# Face recognition settings
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))  # cosine similarity

# Optional: keep enroll images on disk for debugging (MongoDB still stores embeddings ONLY)
SAVE_ENROLL_IMAGES = os.getenv("SAVE_ENROLL_IMAGES", "1") == "1"

FACES_DIR = os.path.join("data", "faces")
os.makedirs(FACES_DIR, exist_ok=True)
