# backend/face_recog_mongo.py
# Requirements: fastapi uvicorn pymongo pillow numpy face_recognition python-multipart
# python -m pip install fastapi uvicorn pymongo pillow numpy face_recognition python-multipart

import io, json, datetime, threading
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Path, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from bson.objectid import ObjectId
from PIL import Image
import numpy as np
import face_recognition
import uvicorn
import os
import uuid

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "fake_attendance")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
students_col = db["students"]
sessions_col = db["attendance_sessions"]  # optional persistent storage

# make sure index on regNo
students_col.create_index([("regNo", ASCENDING)], unique=True)

app = FastAPI(title="FaceRec Mongo Prototype")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory cache of embeddings for fast identification
# cache: list of dicts { regNo, name, embedding (list of floats) }
emb_cache_lock = threading.Lock()
emb_cache: List[Dict[str, Any]] = []

def load_embeddings_into_cache():
    global emb_cache
    with emb_cache_lock:
        emb_cache = []
        for doc in students_col.find({}, {"regNo":1, "name":1, "embeddings":1}):
            reg = doc.get("regNo")
            name = doc.get("name")
            emb_list = doc.get("embeddings", [])
            for emb in emb_list:
                emb_cache.append({"regNo": reg, "name": name, "embedding": emb})
    print(f"[cache] loaded {len(emb_cache)} embeddings")

load_embeddings_into_cache()

# Simple in-memory session store
sessions_lock = threading.Lock()
sessions: Dict[str, Dict[str, Any]] = {}

def read_imagefile_bytes(upload: UploadFile):
    content = upload.file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)

def find_face_encodings(img_np):
    locs = face_recognition.face_locations(img_np, model="hog")
    if not locs:
        return [], []
    encs = face_recognition.face_encodings(img_np, known_face_locations=locs)
    return locs, encs

def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

# --------------------------
# Models (Pydantic if needed)
# --------------------------
class StartSessionResp(BaseModel):
    session_id: str
    created_at: str

# --------------------------
# Endpoints
# --------------------------

@app.post("/session/start", response_model=StartSessionResp)
async def start_session():
    sid = str(uuid.uuid4())
    with sessions_lock:
        sessions[sid] = {
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "identified": [],    # list of {regNo, name, distance, ts}
            "unmatched_count": 0
        }
    return {"session_id": sid, "created_at": sessions[sid]["created_at"]}

@app.post("/upload/face-register")
async def register_student(regNo: str = Form(...), name: str = Form(...), images: List[UploadFile] = File(...)):
    """
    Admin: upload 1..5 images for a student; compute embeddings and store in MongoDB
    """
    all_embeddings = []
    for f in images:
        try:
            img_np = read_imagefile_bytes(f)
        except Exception as e:
            continue
        locs, encs = find_face_encodings(img_np)
        if not encs:
            continue
        # take all encodings found (teacher should upload clean photos)
        for e in encs:
            all_embeddings.append([float(x) for x in e])

    if not all_embeddings:
        raise HTTPException(status_code=400, detail="No faces found in provided images.")

    # upsert student doc
    students_col.update_one(
        {"regNo": regNo},
        {"$set": {"name": name, "updatedAt": datetime.datetime.utcnow()},
         "$push": {"embeddings": {"$each": all_embeddings}}},
        upsert=True
    )

    # reload cache (simple approach)
    load_embeddings_into_cache()

    return {"status": "ok", "regNo": regNo, "name": name, "embeddings_added": len(all_embeddings)}

@app.post("/session/{session_id}/identify")
async def identify(session_id: str = Path(...), frame: UploadFile = File(...), distance_threshold: float = Query(0.55)):
    """
    Main identification endpoint.
    - Accepts a single frame (front-camera)
    - Detect face, compute embedding, compare to all embeddings in cache
    - If best match distance <= threshold -> identified
    - Records into in-memory session
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    try:
        img_np = read_imagefile_bytes(frame)
    except Exception as e:
        raise HTTPException(400, "Invalid image file")

    locs, encs = find_face_encodings(img_np)
    if not encs:
        # no face
        with sessions_lock:
            sessions[session_id]["unmatched_count"] += 1
        return {"identified": False, "reason": "no_face_detected"}

    # For student-by-student usage we expect one face. If multiple, choose largest.
    if len(encs) > 1:
        areas = []
        for (top, right, bottom, left) in locs:
            areas.append((bottom - top) * (right - left))
        idx = int(np.argmax(areas))
        emb = encs[idx]
    else:
        emb = encs[0]

    # compare to cache
    best = None
    best_dist = float("inf")
    with emb_cache_lock:
        for item in emb_cache:
            d = euclidean(emb, item["embedding"])
            if d < best_dist:
                best_dist = d
                best = item

    if best is None:
        return {"identified": False, "reason": "no_db_embeddings"}

    # match decision
    if best_dist <= distance_threshold:
        # record as identified (avoid duplicates in the same session: optional)
        now = datetime.datetime.utcnow().isoformat() + "Z"
        with sessions_lock:
            # avoid duplicate marking if same regNo already in session
            existing_regNos = {x["regNo"] for x in sessions[session_id]["identified"]}
            if best["regNo"] not in existing_regNos:
                sessions[session_id]["identified"].append({"regNo": best["regNo"], "name": best["name"], "distance": best_dist, "ts": now})
        return {"identified": True, "regNo": best["regNo"], "name": best["name"], "distance": best_dist}
    else:
        with sessions_lock:
            sessions[session_id]["unmatched_count"] += 1
        return {"identified": False, "closest_regNo": best["regNo"], "distance": best_dist}

@app.post("/session/{session_id}/finalize")
async def finalize(session_id: str = Path(...), signature_reg_nos: Optional[List[str]] = Body(None)):
    """
    Finalize session. Compare identified list with signature list to find fake signatures.
    Returns present, absent, fake_signatures.
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    with sessions_lock:
        s = sessions[session_id].copy()

    # load all registered students' regNos
    regNos_all = [doc["regNo"] for doc in students_col.find({}, {"regNo": 1})]

    present = [entry["regNo"] for entry in s["identified"]]
    absent = [r for r in regNos_all if r not in present]
    signature_set = set(signature_reg_nos or [])
    fake_signatures = list(signature_set - set(present))

    # optionally store session result in MongoDB
    res_doc = {
        "session_id": session_id,
        "created_at": s["created_at"],
        "present": present,
        "absent": absent,
        "fake_signatures": fake_signatures,
        "identified_records": s["identified"],
        "unmatched_count": s["unmatched_count"],
        "finalized_at": datetime.datetime.utcnow()
    }
    sessions_col.insert_one(res_doc)

    return {
        "session_id": session_id,
        "present": present,
        "absent": absent,
        "fake_signatures": fake_signatures,
        "identified_count": len(present)
    }

@app.get("/students")
async def list_students():
    docs = []
    for d in students_col.find({}, {"regNo":1, "name":1}):
        docs.append({"regNo": d.get("regNo"), "name": d.get("name")})
    return {"students": docs}

@app.post("/admin/reload-embeddings")
async def admin_reload():
    """Admin: reload embeddings cache from DB"""
    load_embeddings_into_cache()
    return {"status": "ok", "loaded": len(emb_cache)}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return sessions[session_id]

if __name__ == "__main__":
    uvicorn.run("face_recog_mongo:app", host="0.0.0.0", port=9000, reload=True)
