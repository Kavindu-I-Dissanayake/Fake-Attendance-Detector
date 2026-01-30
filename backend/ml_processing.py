# ml_processing.py
# Hybrid signature detection: YOLOv8 signature model (yolov8s.pt) + fallback blob heuristic
# Polished for:
#   - more robust row detection (auto-parameter search)
#   - reg_no OCR only (no name OCR; name kept as "" to preserve output schema)
#   - safer debug (no "feats" leakage across rows)
#
# Requirements:
#   pip install ultralytics opencv-python pytesseract numpy
#
# Put 'yolov8s.pt' in same folder (or provide full path)

import os
import cv2
import numpy as np
import pytesseract
import math
import re
from ultralytics import YOLO
from typing import List, Tuple, Dict, Union

# -------------------------
# CONFIG - adjust if needed
# -------------------------
# Your local Tesseract path (Windows)
# Keep this as-is if you use Windows. If the file does not exist, pytesseract will use system PATH.
TESS_PATH = r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# Signature model file
SIGN_MODEL_PATH = "yolov8s.pt"

# -------------------------
# YOLO MODELS
# -------------------------
try:
    signature_model = YOLO(SIGN_MODEL_PATH)
except Exception as e:
    signature_model = None
    print(f"WARNING: Could not load signature model at {SIGN_MODEL_PATH}: {e}")

# -------------------------
# Text cleaning
# -------------------------
def _clean_text_for_reg(text: str) -> str:
    text = text.replace("\\", "/").replace(" ", "")
    return re.sub(r"[^0-9A-Za-z/]", "", text).strip()

# Keep name cleaning for schema compatibility (we won't OCR name)
def _clean_text_for_name(text: str) -> str:
    return re.sub(r"[^A-Za-z.\s]", "", text).strip()

def _fix_alpha(img: np.ndarray) -> np.ndarray:
    """Convert BGRA->BGR if image has alpha channel."""
    if img is None:
        return img
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# =====================================================
# Robust row detection (polished)
# =====================================================
def _detect_table_rows_via_lines(gray: np.ndarray, expected_rows: int = 25) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Detect row ranges using horizontal-line morphology, but with a small parameter search.
    This makes it much more reliable on phone photos (broken / faint lines).
    Returns (ranges, centroids).
    """
    H, W = gray.shape[:2]
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _extract_lines_with_params(horiz_len: int, open_iter: int, dilate_iter: int) -> List[int]:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        detected = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk, iterations=open_iter)
        detected = cv2.dilate(detected, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iter)

        cnts, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # require reasonably wide line segment
            if w >= max(10, int(0.35 * W)):
                ys.append(y + h // 2)
        return sorted(set(ys))

    def _ranges_from_lines(line_ys: List[int]) -> List[Tuple[int, int]]:
        if len(line_ys) < 2:
            return []
        ranges = []
        for i in range(len(line_ys) - 1):
            y_top = line_ys[i]
            y_bottom = line_ys[i + 1]
            y1 = min(max(0, y_top + 2), H - 1)
            y2 = max(min(H - 1, y_bottom - 2), 0)
            if y2 - y1 >= 8:
                ranges.append((y1, y2))
        # trim tiny margins (top/bottom) if present
        if ranges:
            heights = np.array([y2 - y1 for (y1, y2) in ranges], dtype=np.float32)
            med_h = float(np.median(heights)) if len(heights) else 0.0
            if med_h > 0:
                min_h = max(8, int(0.45 * med_h))
                ranges = [r for r in ranges if (r[1] - r[0]) >= min_h]
        return ranges

    # --- parameter search: try a few kernel sizes & iterations, pick closest to expected_rows ---
    best = None  # (score, ranges)
    horiz_divs = [10, 12, 15, 18, 20, 25, 30]
    for div in horiz_divs:
        horiz_len = max(20, W // div)
        for open_iter in (1, 2):
            for dilate_iter in (1, 2, 3):
                line_ys = _extract_lines_with_params(horiz_len, open_iter, dilate_iter)
                ranges = _ranges_from_lines(line_ys)

                if not ranges:
                    continue

                # If more than expected, pick the best contiguous window of expected_rows
                if len(ranges) > expected_rows:
                    heights = np.array([r[1] - r[0] for r in ranges], dtype=np.float32)
                    # choose window with most consistent row heights
                    best_start = 0
                    best_var = float("inf")
                    for s in range(0, len(ranges) - expected_rows + 1):
                        window = heights[s:s + expected_rows]
                        v = float(np.var(window))
                        if v < best_var:
                            best_var = v
                            best_start = s
                    ranges = ranges[best_start:best_start + expected_rows]

                score = abs(len(ranges) - expected_rows)

                # Prefer solutions that are not "too short" (missing many rows)
                if len(ranges) < int(expected_rows * 0.75):
                    score += 3

                if best is None or score < best[0]:
                    best = (score, ranges)

                # Perfect match: stop early
                if score == 0 and len(ranges) == expected_rows:
                    break

    if best is not None and best[1]:
        ranges = best[1]
        centroids = [(r[0] + r[1]) // 2 for r in ranges]
        return ranges, centroids

    # --- final fallback: evenly split the sheet into expected_rows ---
    row_h = max(20, H // expected_rows)
    ranges = [(i * row_h, min(H - 1, (i + 1) * row_h - 1)) for i in range(expected_rows)]
    centroids = [(r[0] + r[1]) // 2 for r in ranges]
    return ranges, centroids

def _extract_centered_band(y1: int, y2: int, band_fraction: float = 0.70) -> Tuple[int, int]:
    row_h = max(4, (y2 - y1))
    bh = max(4, int(round(row_h * band_fraction)))
    mid = (y1 + y2) // 2
    top = max(y1, mid - bh // 2)
    bottom = min(y2, mid + bh // 2)
    return top, bottom

# =====================================================
# Signature heuristic (kept mostly same; small robustness)
# =====================================================
def _analyze_signature_band(band_bgr: np.ndarray) -> Dict[str, Union[int, float]]:
    res = {"ink_ratio": 0.0, "largest_blob": 0, "contours": 0, "edge_strength": 0, "score": 0.0, "pct_width": 0.0}
    if band_bgr is None or band_bgr.size == 0:
        return res

    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    area = th.size
    ink_pixels = int(np.count_nonzero(th))
    ink_ratio = ink_pixels / max(1, area)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(cnts)
    largest_blob = int(max((cv2.contourArea(c) for c in cnts), default=0))

    col_sum_cols = np.sum(th // 255, axis=0) if th.size else np.array([])
    col_thresh = max(1, int(th.shape[0] * 0.03))
    ink_cols = np.where(col_sum_cols >= col_thresh)[0] if col_sum_cols.size else np.array([])
    pct_width = float(len(np.unique(ink_cols))) / float(max(1, th.shape[1])) if th.shape[1] > 0 else 0.0

    edges = cv2.Canny(gray, 40, 120)
    edge_strength = int(np.count_nonzero(edges))

    score = (largest_blob * 0.4) + (edge_strength * 0.28) + (contour_count * 8.0) + (ink_ratio * 700.0)

    res.update({
        "ink_ratio": ink_ratio,
        "largest_blob": largest_blob,
        "contours": contour_count,
        "edge_strength": edge_strength,
        "score": score,
        "pct_width": pct_width
    })
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
# YOLO signature detection utilities (polished extraction)
# =====================================================
def detect_signatures_on_image(img_bgr: np.ndarray, conf_thresh: float = 0.28) -> List[Tuple[int, int, int, int, float]]:
    """
    Run signature_model on an image (BGR).
    Returns list of detections as (x1,y1,x2,y2,conf).
    """
    if signature_model is None:
        return []
    try:
        results = signature_model(img_bgr, imgsz=640, conf=conf_thresh, verbose=False)
    except Exception as e:
        print(f"Signature model inference error: {e}")
        return []

    dets: List[Tuple[int, int, int, int, float]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                dets.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        except Exception:
            # ultra-safe fallback
            for box in boxes:
                vals = np.array(box.xyxy)[0]
                c = float(np.array(box.conf)[0]) if len(np.array(box.conf)) > 0 else 0.0
                dets.append((int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), c))
    return dets

# =====================================================
# OCR for Reg No only (polished)
# =====================================================
def _ocr_reg_no(reg_crop_bgr: np.ndarray) -> str:
    if reg_crop_bgr is None or reg_crop_bgr.size == 0:
        return ""

    g = cv2.cvtColor(reg_crop_bgr, cv2.COLOR_BGR2GRAY)

    # upscale helps tesseract on small printed text
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/"
    try:
        txt = pytesseract.image_to_string(th, config=cfg)
    except Exception:
        txt = ""

    cleaned = _clean_text_for_reg(txt)

    # Try to extract a sensible pattern like 2021/ICT/02 (works well on your sample)
    m = re.search(r"\d{4}/[A-Za-z]{2,10}/\d{1,3}", cleaned)
    return m.group(0) if m else cleaned

# =====================================================
# Main: Hybrid wrapper (same interface, polished internals)
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
      - detect rows via printed lines (robust parameter search)
      - run YOLO once on signature column (optional)
      - for each row: if YOLO intersects row band => signed
                   else fallback to blob heuristic
    Absentees list includes reg_no only (name kept as "" for compatibility).
    """
    print(f"\n[ SIGN-SHEET PROCESSING - HYBRID (YOLO + FallBack) ] -> {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return {"total_students": 0, "present_count": 0, "absent_count": 0, "absentees": [],
                "message": "Could not load image", "debug_image": None}

    img = _fix_alpha(img)
    H, W = img.shape[:2]

    # scale baseline signature column coordinates (signature column is rightmost)
    scale = float(W) / float(max(1, baseline_sig_right_px))
    sig_x1 = max(0, int(round(baseline_sig_left_px * scale)))
    sig_x2 = min(W, int(round(baseline_sig_right_px * scale)))
    sig_w = max(1, sig_x2 - sig_x1)

    # Inner margin to reduce false positives from table borders
    inner_margin = max(2, int(sig_w * 0.03))
    sig_inner_x1 = min(sig_x2 - 1, sig_x1 + inner_margin)
    sig_inner_x2 = max(sig_inner_x1 + 1, sig_x2 - inner_margin)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect rows via robust lines
    row_ranges, _ = _detect_table_rows_via_lines(gray, expected_rows=max_rows)
    if not row_ranges:
        return {"total_students": 0, "present_count": 0, "absent_count": 0, "absentees": [],
                "message": "No rows detected", "debug_image": None}

    if len(row_ranges) > max_rows:
        row_ranges = row_ranges[:max_rows]

    total_rows = len(row_ranges)
    print(f"Detected rows = {total_rows}")

    # Run YOLO once on full signature column
    yolo_detections: List[Tuple[int, int, int, int, float]] = []
    if use_column_yolo and signature_model is not None:
        try:
            sig_col = img[:, sig_x1:sig_x2].copy()
            col_dets = detect_signatures_on_image(sig_col, conf_thresh=yolo_conf)
            for (x1, y1, x2, y2, conf) in col_dets:
                yolo_detections.append((x1 + sig_x1, y1, x2 + sig_x1, y2, conf))
        except Exception as e:
            print(f"YOLO column detect error: {e}")
            yolo_detections = []

    # First pass: compute median blob for fallback baseline (use inner margin)
    blob_list = []
    for (y1, y2) in row_ranges:
        top, bottom = _extract_centered_band(y1, y2, band_fraction=0.70)
        band = img[top:bottom, sig_inner_x1:sig_inner_x2]
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

    # OCR boundaries (keep your existing % approach)
    serial_x2 = int(W * 0.08)
    reg_x1, reg_x2 = serial_x2, int(W * 0.30)

    for idx, (y1, y2) in enumerate(row_ranges, start=1):
        band_top, band_bottom = _extract_centered_band(y1, y2, band_fraction=0.70)
        row_h = max(1, y2 - y1)

        # Always define feats so debug never "leaks" previous row
        feats = {"score": 0.0, "largest_blob": 0, "ink_ratio": 0.0, "pct_width": 0.0}

        # --- YOLO row matching (more strict to avoid cross-row bleed) ---
        signed_by_yolo = False
        best_conf = 0.0
        for (dx1, dy1, dx2, dy2, conf) in yolo_detections:
            det_h = max(1, dy2 - dy1)

            # ignore detections that are unrealistically tall (often span multiple rows)
            if det_h > int(row_h * 2.2):
                continue

            overlap_top = max(band_top, dy1)
            overlap_bot = min(band_bottom, dy2)
            overlap_h = max(0, overlap_bot - overlap_top)

            if overlap_h <= 0:
                continue

            overlap_ratio = overlap_h / float(det_h)
            dcy = (dy1 + dy2) // 2

            # accept if overlap is meaningful or center falls inside band
            if overlap_ratio >= 0.25 or (band_top <= dcy <= band_bottom):
                signed_by_yolo = True
                best_conf = max(best_conf, float(conf))

        # Fallback band (use inner margin)
        band = img[band_top:band_bottom, sig_inner_x1:sig_inner_x2]

        # OCR reg for this row (printed, stable)
        reg_crop = img[y1:y2, reg_x1:reg_x2]
        reg_clean = _ocr_reg_no(reg_crop)

        # name not needed, but keep schema
        name_clean = ""

        if signed_by_yolo and best_conf >= yolo_conf:
            signed = True
        else:
            feats = _analyze_signature_band(band) if band.size else feats
            signed = _decide_signed(
                feats, median_blob,
                ink_cutoff=INK_CUTOFF,
                blob_cutoff_min=BLOB_MIN,
                pct_width_min=PCT_WIDTH_MIN,
                SCORE_THRESHOLD=SCORE_THRESHOLD
            )

        # ---- Debug drawing (same filenames and general style) ----
        colorA = (0, 200, 0) if signed else (0, 0, 255)
        cv2.rectangle(debug_img_A, (sig_x1, y1), (sig_x2, y2), colorA, 1)
        cv2.putText(debug_img_A, f"R{idx}{'P' if signed else 'A'}", (sig_x1 + 4, max(14, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colorA, 1)

        overlay = debug_img_B.copy()
        alpha = 0.18
        cv2.rectangle(overlay, (sig_x1, band_top), (sig_x2, band_bottom), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, debug_img_B, 1 - alpha, 0, debug_img_B)

        cv2.rectangle(debug_img_B, (sig_x1, y1), (sig_x2, y2), (200, 200, 0), 1)
        box_color = (20, 200, 20) if signed else (0, 0, 255)
        text_color = (20, 200, 20) if signed else (0, 215, 255)
        cv2.rectangle(debug_img_B, (sig_x1, band_top), (sig_x2, band_bottom), box_color, 3)

        # show score + (optional) best yolo conf
        sc = int(feats.get("score", 0))
        yconf = best_conf if signed_by_yolo else 0.0
        txt = f"R{idx} {'P' if signed else 'A'} S{sc} Y{yconf:.2f}"

        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx, ty = sig_x1 + 6, max(16, y1 + 16)
        cv2.rectangle(debug_img_B, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(debug_img_B, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        # reg_no overlay
        if reg_clean:
            cv2.putText(debug_img_B, reg_clean, (sig_x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

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
