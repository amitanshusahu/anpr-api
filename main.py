# ======================================
# Lightweight CPU ANPR Pipeline
#
#   Video
#     │
#     ▼
#   YOLO vehicle detection + ByteTrack
#     │
#     ▼
#   ONLY selected tracks (stable, close enough, plate not yet final)
#     │
#     ▼
#   Plate detection every ~10 frames
#     │
#     ▼
#   Quality check
#     │
#     ▼
#   OCR only on good plates (single pass)
#     │
#     ▼
#   Temporal voting
#     │
#     ▼
#   FINAL PLATE
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
PLATE_MODEL = "models/anpr_best.pt"
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WHEELER = {"car": 4, "motorcycle": 2, "bus": 6, "truck": 6}

VEHICLE_CONF = 0.25
PLATE_CONF = 0.30
DET_IMGSZ = 736            # vehicle detection input size (smaller = faster on CPU)

MIN_TRACK_FRAMES = 3       # ignore tracks younger than this
MIN_VEHICLE_H = 48         # px; skip far-away vehicles
PLATE_INTERVAL = 10        # frames between plate detection attempts per track
MIN_PLATE_H = 16           # px; skip tiny plates
MIN_PLATE_QUALITY = 0.20   # quality gate before OCR
CONFIRM_VOTES = 3          # agreeing readings needed to finalize a plate

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_TARGET_H = 96          # upscale plate height to ~this before OCR

PLATE_PAD_X = 0.10         # horizontal pad fraction of plate width when cropping
PLATE_PAD_Y = 0.20         # vertical pad fraction of plate height when cropping


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


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


# --------------------------- OCR ENGINE -------------------------------

class PlateReader:
    """EasyOCR wrapper: ONE grayscale upscaled pass per plate crop."""

    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)

    def read(self, plate_bgr: np.ndarray):
        """Returns (text, conf); empty text when nothing readable."""
        h = plate_bgr.shape[0]
        if h < OCR_TARGET_H:
            f = OCR_TARGET_H / max(h, 1)
            plate_bgr = cv2.resize(plate_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        try:
            raw = self.reader.readtext(img, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
        except Exception:
            return "", 0.0
        if not raw:
            return "", 0.0
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
        return joined.strip(), float(conf)


# ------------------------ TEMPORAL VOTING ------------------------------

class PlateVoter:
    """Exact-match voting over normalized OCR readings for one track."""

    def __init__(self):
        self.votes: dict[str, list] = {}   # text -> [count, conf_sum]
        self.readings: list[dict] = []

    def add(self, frame: int, text: str, conf: float):
        key = normalize_plate(text)
        if not key:
            return
        v = self.votes.setdefault(key, [0, 0.0])
        v[0] += 1
        v[1] += conf
        self.readings.append({"frame": frame, "text": key, "confidence": round(conf, 3)})

    def best(self):
        if not self.votes:
            return None
        text, (count, conf_sum) = max(self.votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        total = sum(v[0] for v in self.votes.values())
        return {
            "text": text,
            "votes": count,
            "total_readings": total,
            "confidence": round((count / total) * (conf_sum / count), 3),
        }

    def confirmed(self) -> bool:
        b = self.best()
        return b is not None and b["votes"] >= CONFIRM_VOTES and b["votes"] * 2 >= b["total_readings"]


# --------------------------- VEHICLE RECORD ---------------------------

@dataclass
class VehicleRecord:
    track_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    hits: int = 0
    last_plate_frame: int = -(10 ** 9)
    voter: PlateVoter = field(default_factory=PlateVoter)
    plate: dict | None = None   # frozen once confirmed


# ------------------------------- PIPELINE ------------------------------

class ANPRPipeline:
    """Vehicle detection + ByteTrack + selected tracks + periodic plate
    detection + quality gate + single-pass OCR + temporal voting."""

    def __init__(self, vehicle_model: str = VEHICLE_MODEL, plate_model: str = PLATE_MODEL,
                 imgsz: int = DET_IMGSZ):
        self.imgsz = imgsz
        self.vehicle_model = YOLO(vehicle_model)
        self.plate_model = YOLO(plate_model)
        self.reader = PlateReader()

    def _detect_plate_in_crop(self, crop: np.ndarray):
        """Best plate box (crop coords) inside a vehicle crop, or None."""
        if crop is None or crop.shape[0] < 16 or crop.shape[1] < 16:
            return None
        res = self.plate_model.predict(crop, verbose=False, conf=PLATE_CONF)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        return boxes[i]

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

        print("Pipeline: vehicle det + ByteTrack -> selected tracks -> plate det /10 frames "
              "-> quality check -> OCR -> temporal voting")

        results = self.vehicle_model.track(
            source=source,
            stream=True,
            persist=True,
            tracker=TRACKER,
            classes=list(VEHICLE_CLASSES),
            conf=VEHICLE_CONF,
            imgsz=self.imgsz,
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

            im_draw = im0.copy()
            ann = Annotator(im_draw, line_width=2)
            active = 0

            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                xyxys = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.int().tolist()

                for tid, box, cls in zip(ids, xyxys, clss):
                    vtype = VEHICLE_CLASSES.get(cls, "vehicle")
                    rec = vehicles.get(tid)
                    if rec is None:
                        rec = VehicleRecord(tid, vtype, frame_idx, frame_idx)
                        vehicles[tid] = rec
                    rec.last_frame = frame_idx
                    rec.hits += 1
                    active += 1

                    # ---- track selection: cheap gates before any heavy work ----
                    due = (frame_idx - rec.last_plate_frame) >= PLATE_INTERVAL
                    selected = (
                        rec.plate is None                 # already final -> skip
                        and rec.hits >= MIN_TRACK_FRAMES  # stable track
                        and (box[3] - box[1]) >= MIN_VEHICLE_H  # close enough
                        and due
                    )

                    plate_label = ""
                    if rec.plate:
                        plate_label = f"{rec.plate['text']} {rec.plate['confidence']:.2f}"

                    if selected:
                        rec.last_plate_frame = frame_idx
                        vx1, vy1, vx2, vy2 = map(int, box)
                        x1, y1 = max(0, vx1), max(0, vy1)
                        x2, y2 = min(W, vx2), min(H, vy2)
                        crop = im0[y1:y2, x1:x2]

                        pbox = self._detect_plate_in_crop(crop)
                        if pbox is not None:
                            px1, py1, px2, py2 = map(int, pbox)
                            pad_x = max(2, int(PLATE_PAD_X * (px2 - px1)))
                            pad_y = max(2, int(PLATE_PAD_Y * (py2 - py1)))
                            plate_crop = crop[
                                max(0, py1 - pad_y):min(crop.shape[0], py2 + pad_y),
                                max(0, px1 - pad_x):min(crop.shape[1], px2 + pad_x),
                            ]
                            q = plate_quality(plate_crop)
                            if (py2 - py1) >= MIN_PLATE_H and q >= MIN_PLATE_QUALITY:
                                text, ocr_conf = self.reader.read(plate_crop)
                                if ocr_conf > 0:
                                    rec.voter.add(frame_idx, text, ocr_conf)
                                    if rec.voter.confirmed():
                                        rec.plate = rec.voter.best()
                                        plate_label = (f"{rec.plate['text']} "
                                                       f"{rec.plate['confidence']:.2f}")
                            # plate box in full-frame coords (this attempt only)
                            ann.box_label(
                                [x1 + px1, y1 + py1, x1 + px2, y1 + py2],
                                "", color=(0, 0, 255),
                            )

                    label = f"#{tid} {vtype}"
                    if plate_label:
                        label += f" | {plate_label}"
                    ann.box_label(box, label, color=colors(tid, True))

            frame = ann.result()
            read = sum(1 for v in vehicles.values() if v.plate)
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

        # tracks that never reached CONFIRM_VOTES still get their best guess
        for rec in vehicles.values():
            if rec.plate is None:
                rec.plate = rec.voter.best()

        payload = self._build_json(source, output_video, fps, frame_idx + 1,
                                   vehicles, time.time() - t0)
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2))

        read = [v for v in vehicles.values() if v.plate]
        print(f"\nDone: {frame_idx + 1} frames in {time.time() - t0:.0f}s")
        print(f"  Vehicles tracked : {len(vehicles)}")
        print(f"  Plates read      : {len(read)}")
        for v in sorted(vehicles.values(), key=lambda v: v.track_id):
            plate = v.plate["text"] if v.plate else "-"
            conf = f"{v.plate['confidence']:.2f}" if v.plate else "  "
            print(f"  #{v.track_id:>3} {v.vehicle_type:<11} plate: {plate:<12} {conf}")
        print(f"\nOutputs: {output_video}\n         {output_json}")

    @staticmethod
    def _build_json(source, output_video, fps, total_frames, vehicles, elapsed):
        entries = []
        for rec in sorted(vehicles.values(), key=lambda v: v.track_id):
            entries.append({
                "track_id": rec.track_id,
                "vehicle_type": rec.vehicle_type,
                "wheeler": WHEELER.get(rec.vehicle_type),
                "frames_seen": rec.hits,
                "first_frame": rec.first_frame,
                "last_frame": rec.last_frame,
                "first_seen_sec": round(rec.first_frame / fps, 2),
                "last_seen_sec": round(rec.last_frame / fps, 2),
                "plate": {
                    "text": rec.plate["text"],
                    "confidence": rec.plate["confidence"],
                    "votes": rec.plate["votes"],
                    "total_readings": rec.plate["total_readings"],
                    "reading_log": rec.voter.readings,
                } if rec.plate else None,
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
            },
            "vehicles": entries,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV vehicle ANPR pipeline (CPU-friendly)")
    parser.add_argument("--source", default="data/test.mp4")
    parser.add_argument("--output-video", default="output/anpr4_output.mp4")
    parser.add_argument("--output-json", default="output/anpr4_output.json")
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
