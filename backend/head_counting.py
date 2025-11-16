# head_counting.py
# Head detection using YOLO11m.pt (high accuracy for crowded lecture halls)

import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load your YOLO11m model
HEAD_MODEL_PATH = "yolo11m.pt"
head_model = YOLO(HEAD_MODEL_PATH)


def extract_head_region(x1, y1, x2, y2, head_ratio=0.38):
    """
    Extract top 38% of a person bounding box (head + shoulders).
    """
    h = y2 - y1
    head_h = int(h * head_ratio)
    return x1, y1, x2, y1 + head_h


def get_head_count_v2(video_path: str) -> dict:
    """
    Frame-by-frame YOLO detection:
      - get person bounding boxes
      - crop head regions
      - count people
      - store counts
    Returns:
      - head_count (median of counts)
      - head_count_max
      - processed_frames
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, math.floor(fps))

    counts = []
    processed = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # YOLO detection
        results = head_model(frame, verbose=False)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Extract head region (not used for counting, but good for future)
                    extract_head_region(x1, y1, x2, y2)
                    person_count += 1

        counts.append(person_count)
        processed += 1
        frame_index += 1

    cap.release()

    if not counts:
        return {"head_count": 0, "head_count_max": 0, "processed_frames": 0}

    # Smart aggregation
    top_k = min(7, len(counts))
    top_vals = sorted(counts, reverse=True)[:top_k]

    head_count_median = int(np.median(top_vals))
    head_count_max = int(max(top_vals))

    return {
        "head_count": head_count_median,
        "head_count_max": head_count_max,
        "processed_frames": processed,
        "raw_counts": counts
    }
