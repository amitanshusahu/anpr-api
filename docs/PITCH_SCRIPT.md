# 2-Min Pitch Video Script — SIH 26127 City-Wide AI Engine

> Duration: 2:00 flat (~280 words at 140-150 wpm)
> Problem: SIH 26127 — City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking + Urban Traffic Analytics
> Stack in video: `docs/PREFINAL.pptx` + `output/anpr4_output.mp4` (from `main.py`) + frontend (powered by `api.py`)

## How to use this file

Each scene has 4 rows. Read top to bottom, shoot in order:

- **ON SCREEN** — what the viewer sees. Fullscreen, no split-screen.
- **SAY** — word-for-word voiceover. Read exactly as written for timing.
- **DO** — what you click / play / point at while saying it.
- **CHECK** — cut only when this is true.

## How to read (delivery guide)

- Pace: calm, 140 wpm. Pause 0.5s at every `...` and 1s at every `[BEAT]`.
- Tone: confident builder, not salesy. Stress CAPITALISED words slightly louder.
- Never read slide text aloud. Paraphrase the slide — SAY and ON SCREEN are intentionally different.
- Record voiceover AFTER screen recording. Lay voiceover on top in edit for clean 2:00.
- If you stumble, re-record only that scene. All scenes are hard-cut friendly.

## Pre-shoot checklist (5 min before)

1. `chmod +x start_api.sh && ./start_api.sh` — API must be live for frontend maps.
2. Open `docs/PREFINAL.pptx` in slideshow mode. Update it first per note at bottom.
3. Have `output/anpr4_output.mp4` ready in VLC, paused at 0:01, fullscreen-ready.
4. Have frontend open in Chrome, logged in, 3 tabs pre-loaded:
   - Tab A: Density heatmap (Brahmapur, 12 cameras)
   - Tab B: Vehicle search / tracking view
   - Tab C: Alerts map
5. Close notifications, set 1080p screen record, mic test.

---

## SCENE 1 — HOOK [0:00-0:15] (15s, ~35 words)

**ON SCREEN:** `PREFINAL.pptx` Slide 1 — Title: City-Wide AI Engine + city CCTV collage.

**SAY:**
> "Cities have THOUSANDS of CCTV cameras... but they work in SILOS. We cannot follow a stolen car across the city... [BEAT] or see where a traffic jam BEGINS. We're building the City-Wide AI Engine that CONNECTS them."

**DO:** Start on title. Slow zoom into CCTV collage on "THOUSANDS". Hard cut on "CONNECTS".

**CHECK:** Viewer understands: silos = problem, connected city = promise.

---

## SCENE 2 — PROBLEM [0:15-0:40] (25s, ~60 words)

**ON SCREEN:** PPTX Slide 2 — 3 asks of SIH 26127 (icons only, no paragraphs).

**SAY:**
> "Problem 26127 asks for THREE things. One — 90 percent-plus ANPR OCR... in light, rain, angle, blur, dirty plates. Two — trajectory tracking... any plate's full ROUTE on a GIS map, with time. Three — macro analytics... density, origin-destination, bottlenecks, HEATMAPS... plus blacklist ALERTS."

**DO:** Point / highlight icon 1, 2, 3 as you say One / Two / Three. Hold 1s on ALERTS.

**CHECK:** All 3 asks are named. Do not explain solution yet.

---

## SCENE 3 — APPROACH [0:40-1:00] (20s, ~50 words)

**ON SCREEN:** PPTX Slide 3 — Pipeline diagram from `docs/Arcitecture.md` Sec #1 + observation JSON snippet (Sec #15).

**SAY:**
> "Our insight... ANPR is NOT the product. Vehicle OBSERVATIONS... plus identity... plus space-time association... IS. YOLO detection, ByteTrack, separate plate YOLO, temporal-vote OCR... plus color, type, make-model, embedding, speed. Every camera emits ONE JSON... to backend... to GIS dashboard."

**DO:** Trace left-to-right on pipeline with cursor: CCTV → YOLO → ByteTrack → OCR → JSON → Map. Pause cursor on JSON snippet on "ONE JSON".

**CHECK:** Say the insight sentence slowly — it's the line judges remember.

---

## SCENE 4 — PROTOTYPE main.py [1:00-1:20] (20s, ~45 words)

**ON SCREEN:** Fullscreen `output/anpr4_output.mp4` (output of `main.py --source data/test.mp4`). No PPTX, no facecam.

**SAY:**
> "This is our WORKING prototype... main.py. Watch — it tracks car 182... runs plate detection only every 10 frames... on STABLE tracks... quality-gates the crop... then OCRs ONCE. Three agreeing votes... FREEZE the final plate."

**DO:** Play video at 1x. Freeze 1s on a clear `#ID car | OD02AB1234 0.96` box. Point at HUD: `Tracks vs Plates read`.

**CHECK:** Plate box + confidence + Track ID must be readable in final export. Re-export from `main.py` if blurry.

---

## SCENE 5 — FRONTEND LIVE (api.py) [1:20-1:50] (30s, ~65 words)

**ON SCREEN:** Your frontend only. No Swagger. 3 quick map demos. Pre-record this as one continuous screen capture, cut dead loading in edit.

### 5A — Density [1:20-1:30]

**SAY:**
> "Now the city BRAIN, live on our api.py. One — DENSITY map... heatmap from /traffic/density... across 12 Brahmapur cameras. Red is congestion... you see bottlenecks INSTANTLY."

**DO:** Tab A. Show heatmap layer ON. Pan across Brahmapur grid once.

### 5B — Track with NO plate [1:30-1:42]

**SAY:**
> "Two — track with NO plate. Filter by white... Swift... CAM_007... time-window. We re-identify by APPEARANCE and embedding... even when the plate is missing."

**DO:** Tab B. Type `color=white, model=Swift, camera=CAM_007`. Show same vehicle polyline across 3+ cameras.

### 5C — Alerts [1:42-1:50]

**SAY:**
> "Three — ALERTS map. Live pins from /traffic/alert... with severity. Blacklist, wrong-way, blockage... in REAL time."

**DO:** Tab C. Click one pin to show popup: `alert_id, lat-lon, severity`.

**CHECK:** All 3 maps shown. If API lags, cut loading, keep voiceover continuous.

---

## SCENE 6 — WALLS + CLOSE [1:50-2:00] (10s, ~25 words)

**ON SCREEN:** PPTX Slide 4 — Roadmap / Walls + team + Thank You.

**SAY:**
> "Walls we'll CROSS... night-rain OCR... cross-camera Re-ID... true speed calibration... RTSP scale. Next — PaddleOCR, pgvector search. THANK YOU."

**DO:** Show 4 roadmap items as checklist. End on Thank You + team names. Stop recording at exactly 2:00.

**CHECK:** End on energy up on "THANK YOU". No new info after 1:58.

---

## Edit order

1. Lay screen recording on timeline first, trim to 2:00.
2. Lay voiceover second, align to scene cuts above.
3. Add soft bg music at -20dB, subtitles burned in.
4. Export 1080p MP4, filename: `SIH26127_2min_pitch.mp4`.

## Update PREFINAL.pptx from Arcitecture.md (do before shoot)

- Slide 2 (Problem): copy the 3 core asks verbatim from `Arcitecture.md` lines 7-8.
- Slide 3 (Architecture): use diagram Sec #1 + identity_score formula Sec #11 + pgvector query Sec #16.
- Slide 4 (Demo UI): use dashboard mock Sec #18 + MOAT milestone (single-cam → multi-cam trajectory CAM-01 → CAM-07 → CAM-12).
- Add 1 line footer on Slide 3: "Temporal OCR aggregation + Re-ID = robustness, not just demo accuracy" (Sec One important correction).
