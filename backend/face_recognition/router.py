from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from uuid import uuid4
from datetime import datetime
import numpy as np
from io import BytesIO

from fastapi.responses import Response
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .config import (
    CLASS_NAME,
    FACES_DIR,
    SAVE_ENROLL_IMAGES,
    FACE_MATCH_THRESHOLD,
)
from .storage import save_student_images
from .db import students_col, sessions_col, attendance_records_col
from .recognition import decode_image_bytes_to_bgr, extract_single_embedding

router = APIRouter(prefix="/face", tags=["Face Recognition"])


def _to_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-10)


@router.post("/enroll")
async def enroll(
    studentId: str = Form(...),
    name: str = Form(...),
    className: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if className != CLASS_NAME:
        raise HTTPException(status_code=400, detail=f"Only '{CLASS_NAME}' supported")

    if len(images) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 images required")

    studentId = studentId.strip()
    name = name.strip()
    if not studentId or not name:
        raise HTTPException(status_code=400, detail="studentId and name are required")

    for img in images:
        if not (img.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="All uploads must be images")

    # --- Extract embeddings from 3 images
    embs = []
    for img in images:
        b = await img.read()
        await img.seek(0)  # allow saving later
        img_bgr = decode_image_bytes_to_bgr(b)
        try:
            emb = extract_single_embedding(img_bgr)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Enroll image error: {str(e)}")
        embs.append(emb)

    avg = _to_unit(np.mean(np.stack(embs, axis=0), axis=0))
    embedding_list = avg.astype(float).tolist()  # store in MongoDB as list[float]

    # Optional: save images to disk (not MongoDB)
    if SAVE_ENROLL_IMAGES:
        save_student_images(FACES_DIR, studentId, images)

    now = datetime.utcnow()

    # Upsert student (MongoDB stores embedding ONLY)
    students_col.update_one(
        {"className": className, "studentId": studentId},
        {
            "$set": {
                "name": name,
                "embedding": embedding_list,
                "updatedAt": now,
            },
            "$setOnInsert": {
                "createdAt": now,
            },
        },
        upsert=True,
    )

    return {
        "message": "enrolled_with_embedding",
        "studentId": studentId,
        "name": name,
        "className": className,
    }


@router.post("/session/start")
def start_session(className: str = Form(...)):
    if className != CLASS_NAME:
        raise HTTPException(status_code=400, detail=f"Only '{CLASS_NAME}' supported")

    sessionId = str(uuid4())
    now = datetime.utcnow()

    sessions_col.insert_one(
        {
            "sessionId": sessionId,
            "className": className,
            "createdAt": now,
            "presentStudentIds": [],
        }
    )

    return {"message": "session_started", "sessionId": sessionId, "className": className}


@router.post("/session/verify")
async def verify_face(
    sessionId: str = Form(...),
    image: UploadFile = File(...)
):
    session = sessions_col.find_one({"sessionId": sessionId})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    className = session["className"]

    # Read scan image
    b = await image.read()
    img_bgr = decode_image_bytes_to_bgr(b)

    try:
        scan_emb = extract_single_embedding(img_bgr)
    except ValueError as e:
        return {"status": "unknown", "message": str(e)}

    # Fetch enrolled embeddings
    enrolled = list(
        students_col.find(
            {"className": className},
            {"_id": 0, "studentId": 1, "name": 1, "embedding": 1},
        )
    )

    if not enrolled:
        return {"status": "unknown", "message": "No enrolled students yet"}

    scan_emb = _to_unit(np.array(scan_emb, dtype=np.float32))

    best = None
    best_score = -1.0

    for s in enrolled:
        emb_list = s.get("embedding")
        if not emb_list:
            continue

        emb = _to_unit(np.array(emb_list, dtype=np.float32))
        score = float(np.dot(scan_emb, emb))  # cosine similarity

        if score > best_score:
            best_score = score
            best = s

    if best is None:
        return {"status": "unknown", "message": "No valid embeddings found"}

    if best_score < FACE_MATCH_THRESHOLD:
        return {
            "status": "unknown",
            "message": f"Face not recognized",
            "score": round(best_score, 3),
            "threshold": FACE_MATCH_THRESHOLD,
        }

    student_id = best["studentId"]
    student_name = best["name"]

    # Mark attendance (no duplicates)
    result = sessions_col.update_one(
        {"sessionId": sessionId, "presentStudentIds": {"$ne": student_id}},
        {"$addToSet": {"presentStudentIds": student_id}},
    )

    if result.modified_count == 0:
        return {
            "status": "already_marked",
            "student": {"studentId": student_id, "name": student_name},
            "message": "Already marked present",
            "score": round(best_score, 3),
        }

    # Optional record history
    attendance_records_col.update_one(
        {"sessionId": sessionId, "studentId": student_id},
        {
            "$setOnInsert": {
                "sessionId": sessionId,
                "studentId": student_id,
                "name": student_name,
                "className": className,
                "createdAt": datetime.utcnow(),
            }
        },
        upsert=True,
    )

    return {
        "status": "verified",
        "student": {"studentId": student_id, "name": student_name},
        "message": "Marked present",
        "score": round(best_score, 3),
    }


@router.get("/session/report/{sessionId}")
def report(sessionId: str):
    session = sessions_col.find_one({"sessionId": sessionId}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    className = session["className"]
    present_ids = set(session.get("presentStudentIds", []))

    all_students = list(
        students_col.find({"className": className}, {"_id": 0, "studentId": 1, "name": 1})
    )

    present = [s for s in all_students if s["studentId"] in present_ids]
    absent = [s for s in all_students if s["studentId"] not in present_ids]

    return {
        "sessionId": sessionId,
        "className": className,
        "createdAt": session.get("createdAt"),
        "presentCount": len(present),
        "absentCount": len(absent),
        "present": present,
        "absent": absent,
    }


@router.get("/session/report/{sessionId}/pdf")
def report_pdf(sessionId: str):
    session = sessions_col.find_one({"sessionId": sessionId}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    className = session["className"]
    present_ids = set(session.get("presentStudentIds", []))

    all_students = list(
        students_col.find({"className": className}, {"_id": 0, "studentId": 1, "name": 1})
    )

    present = [s for s in all_students if s["studentId"] in present_ids]
    absent = [s for s in all_students if s["studentId"] not in present_ids]

    # Build PDF in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Attendance Report (Face Recognition)")
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Class: {className}")
    y -= 16
    c.drawString(50, y, f"Session ID: {sessionId}")
    y -= 16
    c.drawString(50, y, f"Present: {len(present)}    Absent: {len(absent)}")
    y -= 25

    # Present
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Present Students")
    y -= 18
    c.setFont("Helvetica", 11)

    if not present:
        c.drawString(60, y, "- None")
        y -= 16
    else:
        for i, s in enumerate(present, start=1):
            c.drawString(60, y, f"{i}. {s['studentId']} - {s['name']}")
            y -= 16
            if y < 70:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    y -= 10

    # Absent
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Absent Students")
    y -= 18
    c.setFont("Helvetica", 11)

    if not absent:
        c.drawString(60, y, "- None")
        y -= 16
    else:
        for i, s in enumerate(absent, start=1):
            c.drawString(60, y, f"{i}. {s['studentId']} - {s['name']}")
            y -= 16
            if y < 70:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    headers = {
        "Content-Disposition": f'attachment; filename="attendance_{sessionId}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
