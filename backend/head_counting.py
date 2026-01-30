# backend/head_counting.py
# Head counting using YOLOv5 CrowdHuman model copied into backend/yolov5_head_app

import os
import sys
import cv2
import math
import torch
import numpy as np
from collections import Counter

# -----------------------------
# Path to YOLOv5 repo in backend
# -----------------------------
Y5_DIR = os.path.join(os.path.dirname(__file__), "yolov5_head_app")
sys.path.insert(0, Y5_DIR)

from models.experimental import attempt_load
from utils.general import non_max_suppression

# letterbox location differs between versions
try:
    from utils.augmentations import letterbox
except Exception:
    from utils.datasets import letterbox

# -----------------------------
# Config
# -----------------------------
WEIGHTS = os.path.join(Y5_DIR, "weights", "crowdhuman_yolov5m.pt")
IMG_SIZE = 640
CONF_THRES = 0.30
IOU_THRES = 0.45
DEVICE = torch.device("cpu")

MODEL = attempt_load(WEIGHTS, map_location=DEVICE)
MODEL.eval()

print("MODEL NAMES:", getattr(MODEL, "names", None))


HEAD_CLASS_ID = 1  # if needed set to 0/1, otherwise count all detections


def _count_heads_in_frame(frame_bgr) -> int:
    img = letterbox(frame_bgr, IMG_SIZE, stride=int(MODEL.stride.max()), auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(DEVICE).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = MODEL(im)[0]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

    det = pred[0]
    if det is None or len(det) == 0:
        return 0

    if HEAD_CLASS_ID is None:
        return int(len(det))

    cls_ids = det[:, 5].cpu().numpy().astype(int)
    return int(np.sum(cls_ids == HEAD_CLASS_ID))


def get_head_count_v2(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

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

        c = _count_heads_in_frame(frame)
        counts.append(c)
        processed += 1
        frame_index += 1

    cap.release()

    if not counts:
        return {"head_count": 0, "head_count_max": 0, "processed_frames": 0, "raw_counts": []}

    # Final attendance = MOST COMMON value
    head_count_mode = Counter(counts).most_common(1)[0][0]

    return {
        "head_count": int(head_count_mode),
        "head_count_max": int(max(counts)),
        "processed_frames": int(processed),
        "raw_counts": counts
    }
