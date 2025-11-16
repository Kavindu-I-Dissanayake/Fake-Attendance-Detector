import uvicorn
import shutil
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException

# import signature processor (unchanged)
from ml_processing import get_signature_data

# import face router
#from face_recognition.router import router as face_router

# import NEW head detector
from head_counting import get_head_count_v2


app = FastAPI()

# include face router
#Sapp.include_router(face_router)

TEMP_UPLOAD_DIR = tempfile.gettempdir()


@app.post("/upload/video")
async def upload_video(video_file: UploadFile = File(...)):
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        shutil.copyfileobj(video_file.file, temp)
        temp_path = temp.name

    print(f"[VIDEO] Saved temp file: {temp_path}")

    try:
        result = get_head_count_v2(temp_path)

        return {
            "head_count": result["head_count"],
            "head_count_max": result["head_count_max"],
            "processed_frames": result["processed_frames"],
            "message": "Video processed successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Removed temporary file: {temp_path}")


@app.post("/upload/signsheet")
async def upload_sheet(sheet_file: UploadFile = File(...)):
    if not sheet_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    suffix = os.path.splitext(sheet_file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(sheet_file.file, temp)
        temp_path = temp.name

    print(f"[SIGNATURE] Saved temp file: {temp_path}")

    try:
        data = get_signature_data(temp_path)

        return {
            "total_students": data["total_students"],
            "present_count": data["present_count"],
            "absent_count": data["absent_count"],
            "absentees": data["absentees"],
            "message": data["message"],
            "debug_image": data.get("debug_image")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sheet processing failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def root():
    return {"message": "Backend running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
