from .utils import read_imagefile_bytes, get_face_embeddings
from .embeddings_cache import emb_cache, cache_lock
from fastapi import HTTPException
import numpy as np
import datetime

def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

async def identify_student(session_store, session_id, frame_file, threshold=0.55):
    if session_id not in session_store:
        raise HTTPException(404, "Session not found")

    img_np = read_imagefile_bytes(frame_file)
    encs = get_face_embeddings(img_np)

    if not encs:
        session_store[session_id]["unmatched"] += 1
        return {"identified": False, "reason": "no_face"}

    emb = encs[0]  # one student at a time

    # Compare with embeddings cache
    best_student = None
    best_dist = 999

    with cache_lock:
        for item in emb_cache:
            dist = euclidean(emb, item["embedding"])
            if dist < best_dist:
                best_dist = dist
                best_student = item

    if best_student is None:
        return {"identified": False, "reason": "no_students"}

    if best_dist > threshold:
        session_store[session_id]["unmatched"] += 1
        return {"identified": False, "reason": "no_match", "distance": best_dist}

    # Add student to present list
    session_store[session_id]["present"].append(best_student["regNo"])

    return {
        "identified": True,
        "regNo": best_student["regNo"],
        "name": best_student["name"],
        "distance": best_dist
    }
