# ml_processing.py

# Hybrid signature detection: YOLOv8 signature model (yolov8s.pt) + fallback blob heuristic
# Head detection kept unchanged (your original code)
#
# Requirements:
#   pip install ultralytics opencv-python pytesseract numpy scikit-learn
# Place 'yolov8s.pt' in the same folder (or provide full path)
# Tesseract path must be set to your local tesseract.exe (already set below)

import os
import cv2
import numpy as np
import pytesseract
import math
import re
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Union

# -------------------------
# CONFIG - adjust if needed
# -------------------------
# your confirmed Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# signature model file (you placed it as 'yolov8s.pt')
SIGN_MODEL_PATH = "yolov8s.pt"

# -------------------------
# YOLO MODELS
# -------------------------
# Unchanged head-detection model (your original)
model = YOLO("yolo11m.pt")  # keep this exactly as before (head detection)

# Signature detection model (new)
try:
    signature_model = YOLO(SIGN_MODEL_PATH)
except Exception as e:
    signature_model = None
    print(f"WARNING: Could not load signature model at {SIGN_MODEL_PATH}: {e}")

# =====================================================
# HEAD COUNT FUNCTION (UNCHANGED)
# =====================================================
def get_head_count_from_video(video_path: str) -> int:
    """
    Count the maximum number of detected persons in a video.
    This is your original head-detection function — unchanged.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, math.floor(fps))
    max_head_count = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            results = model(frame, verbose=False)
            person_count = 0
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # class 0 = person
                        person_count += 1

            if person_count > max_head_count:
                max_head_count = person_count

            frame_count += 1

    finally:
        cap.release()

    return max_head_count

# =====================================================
# Helper functions (row detection, signature analysis)
# =====================================================

def _fix_alpha(img: np.ndarray) -> np.ndarray:
    """Convert BGRA->BGR if image has alpha channel."""
    if img is None:
        return img
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _clean_text_for_reg(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z/]", "", text).strip()

def _clean_text_for_name(text: str) -> str:
    return re.sub(r"[^A-Za-z.\s]", "", text).strip()

# Grid-based horizontal lines detection (robust for printed tables)
def _detect_table_rows_via_lines(gray: np.ndarray, expected_rows: int = 25) -> Tuple[List[Tuple[int,int]], List[int]]:
    H, W = gray.shape[:2]
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horiz_len = max(20, W // 10)
    horizontal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    detected = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_k, iterations=2)
    detected = cv2.dilate(detected, np.ones((3,3), dtype=np.uint8), iterations=1)
    cnts, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        horiz_len2 = max(10, W // 20)
        horizontal_k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len2, 1))
        detected2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_k2, iterations=2)
        cnts, _ = cv2.findContours(detected2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_ys = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= max(10, 0.3 * W):
            line_ys.append(y + h//2)
    line_ys = sorted(list(set(line_ys)))
    if len(line_ys) < 2:
        # fallback to projection
        proj = np.sum(bw, axis=1).astype(np.float32)
        smooth = cv2.GaussianBlur(proj, (51,51), 0)
        threshold = max(1.0, np.percentile(smooth, 45) * 0.5)
        mask = smooth > threshold
        ranges = []
        start = None
        for y, val in enumerate(mask):
            if val and start is None:
                start = y
            elif not val and start is not None:
                end = y
                if end - start >= 6:
                    ranges.append((start, end))
                start = None
        if start is not None:
            end = H - 1
            if end - start >= 6:
                ranges.append((start, end))
        if not ranges:
            row_h = max(20, H // expected_rows)
            ranges = [(i*row_h, min(H-1, (i+1)*row_h-1)) for i in range(expected_rows)]
        centroids = [(r[0]+r[1])//2 for r in ranges]
        if len(ranges) > expected_rows:
            ranges = ranges[:expected_rows]
            centroids = centroids[:expected_rows]
        return ranges, centroids
    if line_ys[0] > 3:
        line_ys.insert(0, 0)
    if line_ys[-1] < H-3:
        line_ys.append(H-1)
    ranges = []
    for i in range(len(line_ys)-1):
        y_top = line_ys[i]
        y_bottom = line_ys[i+1]
        y1 = min(max(0, y_top + 2), H-1)
        y2 = max(min(H-1, y_bottom - 2), 0)
        if y2 - y1 >= 8:
            ranges.append((y1, y2))
    if len(ranges) > expected_rows:
        step = len(ranges) / expected_rows
        new_ranges = []
        for i in range(expected_rows):
            idx = int(round(i * step))
            idx = min(idx, len(ranges)-1)
            new_ranges.append(ranges[idx])
        ranges = new_ranges
    centroids = [(r[0]+r[1])//2 for r in ranges]
    return ranges, centroids

def _extract_centered_band(y1: int, y2: int, band_fraction: float = 0.70) -> Tuple[int,int]:
    row_h = max(4, (y2 - y1))
    bh = max(4, int(round(row_h * band_fraction)))
    mid = (y1 + y2) // 2
    top = max(y1, mid - bh // 2)
    bottom = min(y2, mid + bh // 2)
    return top, bottom

def _analyze_signature_band(band_bgr: np.ndarray) -> Dict[str, Union[int,float]]:
    res = {"ink_ratio": 0.0, "largest_blob": 0, "contours": 0, "edge_strength": 0, "score": 0.0, "pct_width": 0.0}
    if band_bgr is None or band_bgr.size == 0:
        return res
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    area = th.size
    ink_pixels = int(np.count_nonzero(th))
    ink_ratio = ink_pixels / max(1, area)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(cnts)
    largest_blob = 0
    if contour_count > 0:
        largest_blob = int(max(cv2.contourArea(c) for c in cnts))
    col_sum_cols = np.sum(th // 255, axis=0) if th.size else np.array([])
    col_thresh = max(1, int(th.shape[0] * 0.03))
    ink_cols = np.where(col_sum_cols >= col_thresh)[0] if col_sum_cols.size else np.array([])
    pct_width = float(len(np.unique(ink_cols))) / float(max(1, th.shape[1])) if th.shape[1] > 0 else 0.0
    edges = cv2.Canny(gray, 40, 120)
    edge_strength = int(np.count_nonzero(edges))
    score = (largest_blob * 0.4) + (edge_strength * 0.28) + (contour_count * 8.0) + (ink_ratio * 700.0)
    res.update({"ink_ratio": ink_ratio, "largest_blob": largest_blob, "contours": contour_count, "edge_strength": edge_strength, "score": score, "pct_width": pct_width})
    return res

def _decide_signed(feat: dict, median_blob: float,
                   ink_cutoff: float = 0.015,
                   blob_cutoff_min: int = 18,
                   pct_width_min: float = 0.14,
                   SCORE_THRESHOLD: float = 140.0) -> bool:
    blob = int(feat.get("largest_blob", 0))
    ink = float(feat.get("ink_ratio", 0.0))
    score = float(feat.get("score", 0.0))
    pct_w = float(feat.get("pct_width", 0.0))
    if blob < blob_cutoff_min:
        return False
    if ink < ink_cutoff:
        return False
    if pct_w < pct_width_min:
        if not (score >= SCORE_THRESHOLD * 1.05 and blob >= max(40, int(0.20 * median_blob))):
            return False
    if blob >= max(1, 0.25 * median_blob):
        return True
    if score >= SCORE_THRESHOLD:
        return True
    if (score >= 135.0) and (blob >= 70) and (pct_w >= 0.24):
        return True
    return False

# =====================================================
# YOLO signature detection utilities
# =====================================================

def detect_signatures_on_image(img_bgr: np.ndarray, conf_thresh: float = 0.28) -> List[Tuple[int,int,int,int,float]]:
    """
    Run signature_model on an image (BGR). Returns list of detections
    as (x1,y1,x2,y2,conf) in image coordinates.
    If signature_model failed to load, returns [].
    """
    if signature_model is None:
        return []
    # run inference once
    try:
        results = signature_model(img_bgr, imgsz=640, conf=conf_thresh, verbose=False)
    except Exception as e:
        print(f"Signature model inference error: {e}")
        return []
    dets = []
    for r in results:
        # r.boxes may be an object collection; iterate
        for box in getattr(r, "boxes", []):
            try:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, 'cpu') else float(box.conf[0])
            except Exception:
                # fallback: attempt to read array-like
                vals = np.array(box.xyxy)[0]
                x1, y1, x2, y2 = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
                conf = float(np.array(box.conf)[0]) if len(np.array(box.conf))>0 else 0.0
            dets.append((x1, y1, x2, y2, conf))
    return dets

# =====================================================
# Main Hybrid wrapper: Option C
# =====================================================
def get_signature_data(image_path: str,
                       baseline_sig_left_px: int = 374,
                       baseline_sig_right_px: int = 568,
                       max_rows: int = 25,
                       debug_style: str = "both",
                       use_column_yolo: bool = True,
                       yolo_conf: float = 0.28) -> dict:
    """
    Hybrid signature processing:
      - detect rows via printed lines
      - run YOLO once on signature column (if use_column_yolo True) to get all detections
      - for each row: if YOLO box intersects row band => signed
                   else fallback to blob heuristic
    Returns dict with totals, absentees, and debug image(s).
    debug_style: "A" (raw), "B" (enhanced), "both"
    """
    print(f"\n[ SIGN-SHEET PROCESSING - HYBRID (YOLO + FallBack) ] -> {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return {"total_students": 0, "present_count": 0, "absent_count": 0, "absentees": [], "message": "Could not load image", "debug_image": None}

    img = _fix_alpha(img)
    H, W = img.shape[:2]

    # scale baseline signature column coordinates (from your baseline)
    scale = float(W) / float(max(1, baseline_sig_right_px))
    sig_x1 = max(0, int(round(baseline_sig_left_px * scale)))
    sig_x2 = min(W, int(round(baseline_sig_right_px * scale)))
    sig_w = max(1, sig_x2 - sig_x1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect rows via lines
    row_ranges, centroids = _detect_table_rows_via_lines(gray, expected_rows=max_rows)
    if not row_ranges or len(row_ranges) == 0:
        return {"total_students": 0, "present_count": 0, "absent_count": 0, "absentees": [], "message": "No rows detected", "debug_image": None}

    # limit to max_rows
    if len(row_ranges) > max_rows:
        row_ranges = row_ranges[:max_rows]

    total_rows = len(row_ranges)
    print(f"Detected rows = {total_rows}")

    # Do a full-column YOLO detection once (faster)
    yolo_detections = []
    if use_column_yolo and signature_model is not None:
        try:
            sig_col = img[:, sig_x1:sig_x2].copy()
            # run detection on column; returns boxes relative to sig_col coords
            col_dets = detect_signatures_on_image(sig_col, conf_thresh=yolo_conf)
            # convert to coordinates relative to original image
            for (x1, y1, x2, y2, conf) in col_dets:
                yolo_detections.append((x1 + sig_x1, y1, x2 + sig_x1, y2, conf))
        except Exception as e:
            print(f"YOLO column detect error: {e}")
            yolo_detections = []

    # First pass: compute median blob for fallback baseline
    blob_list = []
    for (y1, y2) in row_ranges:
        top, bottom = _extract_centered_band(y1, y2, band_fraction=0.70)
        band = img[top:bottom, sig_x1:sig_x2]
        feats = _analyze_signature_band(band) if band.size else {"largest_blob": 0}
        blob_list.append(int(feats.get("largest_blob", 0)))
    median_blob = float(max(1.0, np.median(blob_list)))

    # tuned defaults
    SCORE_THRESHOLD = 140.0
    INK_CUTOFF = 0.015
    BLOB_MIN = 18
    PCT_WIDTH_MIN = 0.14

    present_count = 0
    absentees = []
    debug_img_A = img.copy()
    debug_img_B = img.copy()

    # OCR column boundaries (adjust if your sheet differs)
    serial_x1, serial_x2 = 0, int(W * 0.08)
    reg_x1, reg_x2 = serial_x2, int(W * 0.30)
    name_x1, name_x2 = reg_x2, int(W * 0.70)

    # Iterate rows and decide per-row signed status
    for idx, (y1, y2) in enumerate(row_ranges, start=1):
        band_top, band_bottom = _extract_centered_band(y1, y2, band_fraction=0.70)
        # Check YOLO detections: any det whose center Y lies inside band_top..band_bottom
        signed_by_yolo = False
        best_conf = 0.0
        for (dx1, dy1, dx2, dy2, conf) in yolo_detections:
            dcy = (dy1 + dy2) // 2
            # check if vertical overlap significant:
            overlap_top = max(band_top, dy1)
            overlap_bot = min(band_bottom, dy2)
            overlap_h = max(0, overlap_bot - overlap_top)
            det_h = max(1, dy2 - dy1)
            # consider it an intersect if vertical overlap >= 20% of det_h or intersects band center
            if overlap_h >= 0.2 * det_h or (band_top <= dcy <= band_bottom):
                signed_by_yolo = True
                best_conf = max(best_conf, conf)
        # Crop band for fallback on YOLO-miss
        band = img[band_top:band_bottom, sig_x1:sig_x2]
        # OCR fields
        reg_crop = img[y1:y2, reg_x1:reg_x2]
        name_crop = img[y1:y2, name_x1:name_x2]
        try:
            reg_text_raw = pytesseract.image_to_string(reg_crop, config="--psm 7")
        except Exception:
            reg_text_raw = ""
        try:
            name_text_raw = pytesseract.image_to_string(name_crop, config="--psm 7")
        except Exception:
            name_text_raw = ""
        reg_clean = _clean_text_for_reg(reg_text_raw)
        name_clean = _clean_text_for_name(name_text_raw)

        if signed_by_yolo and best_conf >= yolo_conf:
            signed = True
        else:
            # fallback to heuristic
            feats = _analyze_signature_band(band) if band.size else {"largest_blob":0, "score":0.0, "ink_ratio":0.0, "pct_width":0.0}
            signed = _decide_signed(feats, median_blob, ink_cutoff=INK_CUTOFF, blob_cutoff_min=BLOB_MIN, pct_width_min=PCT_WIDTH_MIN, SCORE_THRESHOLD=SCORE_THRESHOLD)

        # Debug drawing: A (raw) and B (enhanced)
        colorA = (0,200,0) if signed else (0,0,255)
        cv2.rectangle(debug_img_A, (sig_x1, y1), (sig_x2, y2), colorA, 1)
        cv2.putText(debug_img_A, f"R{idx}{'P' if signed else 'A'}", (sig_x1+4, max(14, y1+14)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, colorA, 1)

        # Enhanced debug B
        if signed:
            box_color = (20,200,20)
            text_color = (20,200,20)
        else:
            box_color = (0,0,255)
            text_color = (0,215,255)  # yellow-ish
        overlay = debug_img_B.copy()
        alpha = 0.18
        cv2.rectangle(overlay, (sig_x1, band_top), (sig_x2, band_bottom), (0,0,0), -1)
        cv2.addWeighted(overlay, alpha, debug_img_B, 1 - alpha, 0, debug_img_B)
        cv2.rectangle(debug_img_B, (sig_x1, y1), (sig_x2, y2), (200,200,0), 1)
        cv2.rectangle(debug_img_B, (sig_x1, band_top), (sig_x2, band_bottom), box_color, 3)
        txt = f"R{idx} {'P' if signed else 'A'} S{int(feats.get('score',0)) if 'feats' in locals() else 0}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = sig_x1 + 6, max(16, y1 + 16)
        cv2.rectangle(debug_img_B, (tx-2, ty-th-4), (tx+tw+2, ty+2), (0,0,0), -1)
        cv2.putText(debug_img_B, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(debug_img_B, reg_clean if reg_clean else "", (sig_x1+6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        cv2.putText(debug_img_B, name_clean if name_clean else "", (sig_x1+6, y2 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        if signed:
            present_count += 1
        else:
            absentees.append({"serial": str(idx), "reg_no": reg_clean, "name": name_clean})

    absent_count = total_rows - present_count

    # Save debug images according to debug_style
    saved_paths = []
    if debug_style in ("A", "both"):
        pathA = os.path.abspath("debug_sheet_debug_A.png")
        cv2.imwrite(pathA, debug_img_A)
        saved_paths.append(pathA)
    if debug_style in ("B", "both"):
        pathB = os.path.abspath("debug_sheet_debug_B.png")
        cv2.imwrite(pathB, debug_img_B)
        saved_paths.append(pathB)

    debug_image_field = saved_paths[0] if len(saved_paths) == 1 else saved_paths

    print(f"Detected rows = {total_rows}, present={present_count}, absent={absent_count}")
    return {
        "total_students": total_rows,
        "present_count": present_count,
        "absent_count": absent_count,
        "absentees": absentees,
        "message": "Signature sheet processed successfully.",
        "debug_image": debug_image_field
    }