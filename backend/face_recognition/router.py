from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from .register import register_student
from .identify import identify_student
from pymongo import MongoClient
import uuid
import datetime

router = APIRouter(prefix="/face", tags=["Face Recognition"])

# In-memory session store
session_store = {}

@router.post("/register")
async def register(regNo: str = Form(...), name: str = Form(...), images: List[UploadFile] = File(...)):
    return await register_student(regNo, name, images)

@router.post("/session/start")
async def start_session():
    sid = str(uuid.uuid4())
    session_store[sid] = {
        "createdAt": datetime.datetime.utcnow().isoformat(),
        "present": [],
        "unmatched": 0
    }
    return {"session_id": sid}

@router.post("/identify")
async def identify(session_id: str = Form(...), frame: UploadFile = File(...)):
    return await identify_student(session_store, session_id, frame)

@router.post("/session/finalize")
async def finalize(session_id: str = Form(...), signature_list: List[str] = Form([])):
    sess = session_store.get(session_id)
    if not sess:
        return {"error": "Invalid session"}

    present = sess["present"]

    client = MongoClient("mongodb://localhost:27017")
    all_reg_nos = [doc["regNo"] for doc in client["fake_attendance"]["students"].find({}, {"regNo": 1})]

    absent = [r for r in all_reg_nos if r not in present]
    fake_signatures = list(set(signature_list) - set(present))

    return {
        "present": present,
        "absent": absent,
        "fake_signatures": fake_signatures
    }
