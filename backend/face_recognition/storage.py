import os
import shutil
from typing import List
from fastapi import UploadFile

def save_student_images(faces_dir: str, student_id: str, images: List[UploadFile]) -> List[str]:
    student_dir = os.path.join(faces_dir, student_id)
    os.makedirs(student_dir, exist_ok=True)

    saved_paths = []
    for i, img in enumerate(images):
        path = os.path.join(student_dir, f"{i+1}.jpg")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_paths.append(path)

    return saved_paths
