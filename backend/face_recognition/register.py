from .utils import read_imagefile_bytes, get_face_embeddings
from .embeddings_cache import students_col, load_cache
from fastapi import HTTPException
import datetime

async def register_student(regNo, name, images):
    embeddings = []

    # Process all uploaded images
    for img_file in images:
        img_np = read_imagefile_bytes(img_file)
        encs = get_face_embeddings(img_np)

        for e in encs:
            embeddings.append([float(x) for x in e])

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid faces detected.")

    # Insert / update student in DB
    students_col.update_one(
        {"regNo": regNo},
        {
            "$set": {"name": name, "updatedAt": datetime.datetime.utcnow()},
            "$push": {"embeddings": {"$each": embeddings}}
        },
        upsert=True
    )

    # Refresh cache
    load_cache()

    return {
        "status": "ok",
        "regNo": regNo,
        "name": name,
        "embeddings_added": len(embeddings)
    }
