from pymongo import MongoClient
import threading

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "fake_attendance"

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
students_col = db["students"]

emb_cache = []
cache_lock = threading.Lock()

def load_cache():
    """
    Load all stored student embeddings from MongoDB → memory.
    This makes identification fast.
    """
    global emb_cache
    with cache_lock:
        emb_cache = []
        for doc in students_col.find({}, {"regNo": 1, "name": 1, "embeddings": 1}):
            regNo = doc["regNo"]
            name = doc["name"]
            for emb in doc.get("embeddings", []):
                emb_cache.append({
                    "regNo": regNo,
                    "name": name,
                    "embedding": emb
                })
    print(f"[FaceRec] Loaded {len(emb_cache)} embeddings into memory")

# load cache on import
load_cache()
