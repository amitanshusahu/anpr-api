# ======================================
# CCTV Vehicle ANPR Pipeline
#
#   CCTV
#     │
#     ▼
# Vehicle detection (YOLO)
#     │
#     ▼
# ByteTrack  ── Track #41 / #42 / ...
#     │
#     ▼
# Plate detection (per-vehicle crop)
#     │
#     ▼
# Plate crop ──► Quality score + Perspective correction
#     │
#     ▼
# OCR (multiple pre-processed variants)
#     │
#     ▼
# Temporal aggregation (weighted voting across frames)
#     │
#     ▼
# Indian plate validation ──► FINAL PLATE
#
# Outputs: annotated video + JSON list of vehicles with plates
# ======================================

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ----------------------------- CONFIG ---------------------------------

VEHICLE_MODEL = "yolo11n.pt"  # COCO vehicle detector (auto-downloads)
PLATE_MODEL = "models/license-plate-finetune-v1m.pt"
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WHEELER = {"car": 4, "motorcycle": 2, "bus": 6, "truck": 6}

VEHICLE_CONF = 0.25
PLATE_CONF = 0.30

OCR_INTERVAL = 5        # frames between OCR attempts per track
MIN_PLATE_QUALITY = 0.10  # skip OCR on hopeless crops
QUALITY_JUMP = 0.15     # re-run OCR early if quality improved notably
MIN_PLATE_H = 14        # min plate height (px) to attempt OCR

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_TARGET_H = 96       # upscale plate height to ~this before OCR

PLATE_PAD_X = 0.10      # horizontal pad fraction of plate width when cropping
PLATE_PAD_Y = 0.20      # vertical pad fraction of plate height when cropping

# --------------------- INDIAN PLATE VALIDATION ------------------------

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CH", "CG", "GA", "GJ", "HR", "HP", "JH", "JK",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB", "DL", "AN", "DN",
    "DD", "LD", "BH", "LA",
}

# STATE(2) RTO(1-2) LETTERS(1-3) DIGITS(1-4)  e.g. MH12AB1234, DL8CAF4567
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")

# common OCR confusions, fixed per section type
_LETTER_FIX = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"}
_DIGIT_FIX = {"O": "0", "I": "1", "Q": "0", "S": "5", "B": "8", "Z": "2", "G": "6", "D": "0"}


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def correct_indian_plate(text: str):
    """Fit a raw OCR string to the Indian plate pattern, fixing digit/letter
    confusions inside each section. Returns (corrected, num_fixes, format_valid)."""
    t = normalize_plate(text)
    if not 5 <= len(t) <= 11:
        return t, 99, False
    best = (t, 99, False)
    for rto_len in (2, 1):
        for let_len in (1, 2, 3):
            rest_len = len(t) - 2 - rto_len - let_len
            if not 1 <= rest_len <= 4:
                continue
            sections = (
                (t[:2], "L"),
                (t[2:2 + rto_len], "D"),
                (t[2 + rto_len:2 + rto_len + let_len], "L"),
                (t[2 + rto_len + let_len:], "D"),
            )
            out, fixes = [], 0
            for seg, kind in sections:
                for ch in seg:
                    if kind == "D" and not ch.isdigit():
                        rep = _DIGIT_FIX.get(ch)
                        fixes += 1 if rep else 2
                        ch = rep or ch
                    elif kind == "L" and not ch.isalpha():
                        rep = _LETTER_FIX.get(ch)
                        fixes += 1 if rep else 2
                        ch = rep or ch
                    out.append(ch)
            cand = "".join(out)
            valid = bool(INDIAN_PLATE_RE.match(cand))
            score = fixes + (0 if valid else 10)
            if score < best[1]:
                best = (cand, fixes, valid)
    return best


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ------------------------ PLATE IMAGE PROCESSING -----------------------

def plate_quality(roi: np.ndarray) -> float:
    """Heuristic crop quality in [0, 1]: sharpness + contrast + size."""
    h, w = roi.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 200.0)
    contrast = min(1.0, float(gray.std()) / 55.0)
    size = min(1.0, h / 28.0) * min(1.0, w / 90.0)
    return 0.45 * sharp + 0.30 * contrast + 0.25 * size


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def perspective_correct(roi: np.ndarray) -> np.ndarray:
    """Best-effort deskew: minAreaRect over the ink pixels, warped flat.
    Falls back to the original crop when the quad looks unreliable."""
    h, w = roi.shape[:2]
    if h < 10 or w < 24:
        return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.bitwise_not(bw) if (bw == 0).mean() <= 0.5 else bw
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 8]
    if not big:
        return roi
    pts = np.vstack(big).squeeze(1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (rw, rh) = rect[1]
    if rw < 4 or rh < 4:
        return roi
    src = _order_corners(cv2.boxPoints(rect).astype(np.float32))
    wA = np.linalg.norm(src[0] - src[1])
    wB = np.linalg.norm(src[2] - src[3])
    hA = np.linalg.norm(src[0] - src[3])
    hB = np.linalg.norm(src[1] - src[2])
    W, H = int(max(wA, wB)), int(max(hA, hB))
    if W < 10 or H < 6 or not 1.5 <= W / H <= 6.0:
        return roi
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        roi, M, (W, H), flags=cv2.INTER_CUBIC,
        borderValue=tuple(int(c) for c in roi.mean(axis=(0, 1))),
    )


# --------------------------- OCR ENGINE -------------------------------

class PlateReader:
    """EasyOCR wrapper: perspective-corrected + multiple pre-processed
    variants of the plate crop, returning deduplicated (text, conf) readings."""

    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)

    def _variants(self, plate_bgr: np.ndarray) -> list:
        corrected = perspective_correct(plate_bgr)
        bases = [plate_bgr] if corrected is plate_bgr else [plate_bgr, corrected]
        outs = []
        for base in bases:
            h = base.shape[0]
            up = base
            if h < OCR_TARGET_H:
                f = OCR_TARGET_H / max(h, 1)
                up = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            outs.append(up)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            outs.append(clahe)
            _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            outs.append(otsu)
        return outs

    def read(self, plate_bgr: np.ndarray) -> list:
        found = {}
        for img in self._variants(plate_bgr):
            try:
                raw = self.reader.readtext(
                    img, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST
                )
            except Exception:
                continue
            if not raw:
                continue
            frags = []
            for bbox, text, conf in raw:
                if not text:
                    continue
                ys = [p[1] for p in bbox]
                xs = [p[0] for p in bbox]
                frags.append((float(np.mean(ys)), float(np.mean(xs)), text, float(conf)))
            # join fragments row by row (handles two-line plates)
            row_h = img.shape[0] / 3.0
            frags.sort(key=lambda f: (int(f[0] // max(row_h, 1)), f[1]))
            joined = "".join(f[2] for f in frags)
            chars = sum(len(f[2]) for f in frags)
            conf = sum(f[3] * len(f[2]) for f in frags) / max(chars, 1)
            key = normalize_plate(joined)
            if key and (key not in found or conf > found[key][1]):
                found[key] = (joined.strip(), conf)
        return list(found.values())


# ------------------------ TEMPORAL AGGREGATION -------------------------

@dataclass
class Reading:
    frame: int
    text: str
    ocr_conf: float
    quality: float


@dataclass
class PlateResult:
    text: str
    confidence: float
    format_valid: bool
    votes: int
    total_readings: int


class PlateAggregator:
    """Collects OCR readings for one track and produces a consensus plate."""

    def __init__(self):
        self.readings: list[Reading] = []

    def add(self, frame: int, text: str, ocr_conf: float, quality: float):
        norm = normalize_plate(text)
        if norm:
            self.readings.append(Reading(frame, norm, float(ocr_conf), float(quality)))

    def consensus(self):
        if not self.readings:
            return None
        corrected = []
        for r in self.readings:
            text, fixes, _ = correct_indian_plate(r.text)
            if not text:
                continue
            weight = (
                r.ocr_conf
                * (0.5 + 0.5 * r.quality)
                * (1.0 - 0.05 * min(fixes, 5))
            )
            corrected.append((text, r.ocr_conf, weight))
        if not corrected:
            return None
        corrected.sort(key=lambda c: c[2], reverse=True)
        # fuzzy grouping: readings within edit distance 1 vote together
        groups = []  # [text, weight, conf_sum, count]
        for text, conf, weight in corrected:
            for g in groups:
                if levenshtein(text, g[0]) <= 1:
                    g[1] += weight
                    g[2] += conf
                    g[3] += 1
                    break
            else:
                groups.append([text, weight, conf, 1])
        total_w = sum(g[1] for g in groups)
        best = max(groups, key=lambda g: (g[1], g[2]))
        support = best[1] / total_w
        mean_conf = best[2] / best[3]
        valid = bool(INDIAN_PLATE_RE.match(best[0]))
        state_ok = best[0][:2] in STATE_CODES
        conf = support * (0.55 + 0.45 * mean_conf)
        if valid and state_ok:
            conf = min(0.99, conf * 1.15)
        elif valid:
            conf = min(0.99, conf * 1.05)
        else:
            conf *= 0.75
        return PlateResult(best[0], round(conf, 3), valid, best[3], len(self.readings))


# --------------------------- VEHICLE RECORD ---------------------------

@dataclass
class VehicleRecord:
    track_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    det_confs: list = field(default_factory=list)
    aggregator: PlateAggregator = field(default_factory=PlateAggregator)
    last_ocr_frame: int = -(10 ** 9)
    best_quality: float = 0.0
    best_plate: PlateResult = None

    def update_plate(self):
        res = self.aggregator.consensus()
        if res and (self.best_plate is None or res.confidence >= self.best_plate.confidence):
            self.best_plate = res
        return self.best_plate


# ------------------------------- PIPELINE ------------------------------

class ANPRPipeline:
    """Vehicle detection + ByteTrack + plate detection + quality/perspective
    + OCR + temporal aggregation + Indian plate validation."""

    def __init__(self, vehicle_model: str = VEHICLE_MODEL, plate_model: str = PLATE_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vehicle_model = YOLO(vehicle_model)
        self.plate_model = YOLO(plate_model)
        self.reader = PlateReader()

    def _detect_plate_in_crop(self, crop: np.ndarray):
        """Best plate box (crop coords) inside a vehicle crop, or None."""
        if crop is None or crop.shape[0] < 16 or crop.shape[1] < 16:
            return None, 0.0
        res = self.plate_model.predict(crop, verbose=False, conf=PLATE_CONF)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, 0.0
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        return boxes[i], float(confs[i])

    def process_video(
        self,
        source: str,
        output_video: str,
        output_json: str,
        display: bool = False,
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        vehicles: dict[int, VehicleRecord] = {}
        writer = None
        frame_idx = -1
        t0 = time.time()

        print("Pipeline: vehicle det -> ByteTrack -> plate det -> quality/perspective "
              "-> OCR -> temporal aggregation -> Indian validation")

        results = self.vehicle_model.track(
            source=source,
            stream=True,
            persist=True,
            tracker=TRACKER,
            classes=list(VEHICLE_CLASSES),
            conf=VEHICLE_CONF,
            verbose=False,
        )

        for res in results:
            frame_idx += 1
            im0 = res.orig_img
            H, W = im0.shape[:2]

            if writer is None:
                Path(output_video).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
                )

            im_draw = im0.copy()  # keep im0 pristine for crops/OCR
            ann = Annotator(im_draw, line_width=2)
            active = 0

            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                xyxys = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.int().tolist()
                confs = res.boxes.conf.cpu().numpy()

                for tid, box, cls, conf in zip(ids, xyxys, clss, confs):
                    vtype = VEHICLE_CLASSES.get(cls, "vehicle")
                    rec = vehicles.get(tid)
                    if rec is None:
                        rec = VehicleRecord(tid, vtype, frame_idx, frame_idx)
                        vehicles[tid] = rec
                    rec.last_frame = frame_idx
                    rec.det_confs.append(float(conf))
                    active += 1

                    vx1, vy1, vx2, vy2 = map(int, box)
                    cv1, cv2_ = max(0, vx1), min(W, vx2)
                    cy1, cy2 = max(0, vy1), min(H, vy2)
                    crop = im0[cy1:cy2, cv1:cv2_]

                    pbox, pconf = self._detect_plate_in_crop(crop)
                    plate_label = ""
                    if pbox is not None:
                        px1, py1, px2, py2 = map(int, pbox)
                        pad_x = max(2, int(PLATE_PAD_X * (px2 - px1)))
                        pad_y = max(2, int(PLATE_PAD_Y * (py2 - py1)))
                        plate_crop = crop[
                            max(0, py1 - pad_y):min(crop.shape[0], py2 + pad_y),
                            max(0, px1 - pad_x):min(crop.shape[1], px2 + pad_x),
                        ]
                        q = plate_quality(plate_crop)
                        due = (frame_idx - rec.last_ocr_frame) >= OCR_INTERVAL
                        jump = q >= rec.best_quality + QUALITY_JUMP and (frame_idx - rec.last_ocr_frame) >= 2
                        if (due or jump) and q >= MIN_PLATE_QUALITY and (py2 - py1) >= MIN_PLATE_H:
                            for text, ocr_conf in self.reader.read(plate_crop):
                                rec.aggregator.add(frame_idx, text, ocr_conf, q)
                            rec.last_ocr_frame = frame_idx
                            rec.update_plate()
                        rec.best_quality = max(rec.best_quality, q)
                        # plate box in full-frame coords
                        ann.box_label(
                            [cv1 + px1, cy1 + py1, cv1 + px2, cy1 + py2],
                            "", color=(0, 0, 255),
                        )
                        if rec.best_plate:
                            plate_label = f"{rec.best_plate.text} {rec.best_plate.confidence:.2f}"

                    label = f"#{tid} {vtype}"
                    if plate_label:
                        label += f" | {plate_label}"
                    ann.box_label(box, label, color=colors(tid, True))

            frame = ann.result()
            read = sum(1 for v in vehicles.values() if v.best_plate)
            hud = f"Frame {frame_idx} | Tracks: {active} | Plates: {read}"
            cv2.rectangle(frame, (0, 0), (430, 26), (0, 0, 0), -1)
            cv2.putText(frame, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("ANPR pipeline (Press 'q' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}: {len(vehicles)} tracks, {read} plates read "
                      f"({time.time() - t0:.0f}s elapsed)")

        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        for rec in vehicles.values():
            rec.update_plate()

        payload = self._build_json(source, output_video, fps, frame_idx + 1,
                                    vehicles, time.time() - t0)
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2))

        read = [v for v in vehicles.values() if v.best_plate]
        print(f"\nDone: {frame_idx + 1} frames in {time.time() - t0:.0f}s")
        print(f"  Vehicles tracked : {len(vehicles)}")
        print(f"  Plates read      : {len(read)}")
        for v in sorted(vehicles.values(), key=lambda v: v.track_id):
            plate = v.best_plate.text if v.best_plate else "-"
            conf = f"{v.best_plate.confidence:.2f}" if v.best_plate else "  "
            print(f"  #{v.track_id:>3} {v.vehicle_type:<11} plate: {plate:<12} {conf}")
        print(f"\nOutputs: {output_video}\n         {output_json}")

    @staticmethod
    def _build_json(source, output_video, fps, total_frames, vehicles, elapsed):
        entries = []
        for rec in sorted(vehicles.values(), key=lambda v: v.track_id):
            plate = None
            if rec.best_plate:
                pr = rec.best_plate
                plate = {
                    "text": pr.text,
                    "confidence": pr.confidence,
                    "format_valid": pr.format_valid,
                    "state_code": pr.text[:2] if pr.format_valid else None,
                    "num_readings": pr.total_readings,
                    "reading_log": [
                        {
                            "frame": r.frame,
                            "text": r.text,
                            "ocr_confidence": round(r.ocr_conf, 3),
                            "quality": round(r.quality, 3),
                        }
                        for r in rec.aggregator.readings
                    ],
                }
            entries.append({
                "track_id": rec.track_id,
                "vehicle_type": rec.vehicle_type,
                "wheeler": WHEELER.get(rec.vehicle_type),
                "detection_confidence": round(sum(rec.det_confs) / len(rec.det_confs), 3),
                "frames_seen": len(rec.det_confs),
                "first_frame": rec.first_frame,
                "last_frame": rec.last_frame,
                "first_seen_sec": round(rec.first_frame / fps, 2),
                "last_seen_sec": round(rec.last_frame / fps, 2),
                "plate": plate,
            })
        with_plate = [e for e in entries if e["plate"]]
        return {
            "source": source,
            "output_video": output_video,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "fps": fps,
            "total_frames": total_frames,
            "processing_time_sec": round(elapsed, 1),
            "summary": {
                "vehicles_tracked": len(entries),
                "vehicles_with_plate": len(with_plate),
                "plates_valid_format": sum(
                    1 for e in with_plate if e["plate"]["format_valid"]
                ),
            },
            "vehicles": entries,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV vehicle ANPR pipeline")
    parser.add_argument("--source", default="data/test2.mp4")
    parser.add_argument("--output-video", default="output/anpr_output.mp4")
    parser.add_argument("--output-json", default="output/anpr_output.json")
    parser.add_argument("--display", action="store_true", help="show live preview")
    args = parser.parse_args()

    pipeline = ANPRPipeline(
        vehicle_model=VEHICLE_MODEL,
        plate_model=PLATE_MODEL,
    )
    pipeline.process_video(
        source=args.source,
        output_video=args.output_video,
        output_json=args.output_json,
        display=args.display,
    )
