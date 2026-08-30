"""
Mock traffic-monitoring backend.

Serves pre-generated multi-camera vehicle observations from
mock_traffic_data.json (see generate_mock_data.py for how the dataset was
built). Every observation follows the schema:

    camera_id, timestamp, latitude, longitude,
    plate.{text, confidence},
    vehicle.{type, color, make, model, confidence, wheeler},
    motion.{speed_kmh, direction},
    tracking.{local_track_id, embedding},
    bbox [x1, y1, x2, y2]

Endpoints
---------
GET /traffic/density  -> the full observation array (no transformation)
GET /traffic/filter   -> observations filtered by optional AND-ed query params
GET /traffic/alert    -> a random traffic alert near the camera network

Run with:  uvicorn api:app --reload
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_FILE = Path(__file__).parent / "mock_traffic_data.json"

# ---------------------------------------------------------------------------
# Load the mock dataset once at startup.
# ---------------------------------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as _f:
    TRAFFIC_DATA: list[dict] = json.load(_f)

# Camera locations, derived from the observations themselves so the alert
# endpoint stays near the road network even if the dataset is regenerated.
CAMERA_LOC: dict[str, tuple[float, float]] = {}
for _obs in TRAFFIC_DATA:
    CAMERA_LOC.setdefault(_obs["camera_id"], (_obs["latitude"], _obs["longitude"]))

# Distinct make/model values, for validation of filter params.
VALID_MAKES = sorted({o["vehicle"]["make"] for o in TRAFFIC_DATA})
VALID_MODELS = sorted({o["vehicle"]["model"] for o in TRAFFIC_DATA})
VALID_TYPES = sorted({o["vehicle"]["type"] for o in TRAFFIC_DATA})
VALID_COLORS = sorted({o["vehicle"]["color"] for o in TRAFFIC_DATA})
VALID_WHEELERS = sorted({o["vehicle"]["wheeler"] for o in TRAFFIC_DATA})

VALID_ALERTS = [
    "Heavy traffic detected",
    "Traffic congestion",
    "Vehicle stopped",
    "Accident reported",
    "Slow moving traffic",
    "Wrong-way vehicle detected",
    "Road blockage detected",
]
VALID_SEVERITIES = ["low", "medium", "high", "critical"]

app = FastAPI(
    title="Mock Traffic Monitoring API",
    description="Multi-camera vehicle tracking mock backend (Brahmapur, Odisha).",
    version="1.0.0",
)

# Allow any frontend origin to consume this mock API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models.
# ---------------------------------------------------------------------------

class AlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    latitude: float
    longitude: float
    alert: str
    severity: str


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------

@app.get("/traffic/density", response_model=list[dict])
def get_traffic_density():
    """Return every observation, unmodified."""
    return TRAFFIC_DATA


@app.get("/traffic/filter", response_model=list[dict])
def filter_traffic(
    plate: str | None = Query(None, description="Exact plate text (e.g. OD02AB1234)"),
    type: str | None = Query(None, description="Vehicle type (e.g. car, bus, truck)"),
    color: str | None = Query(None, description="Vehicle color (e.g. white, red)"),
    speed: float | None = Query(None, ge=0, description="Exact speed in km/h"),
    min_speed: float | None = Query(None, ge=0, description="Minimum speed in km/h"),
    max_speed: float | None = Query(None, ge=0, description="Maximum speed in km/h"),
    wheeler: int | None = Query(None, ge=2, le=6, description="Wheel count"),
    make: str | None = Query(None, description="Vehicle make (e.g. Maruti Suzuki)"),
    model: str | None = Query(None, description="Vehicle model (e.g. Swift)"),
    camera_id: str | None = Query(None, description="Camera id (e.g. CAM_001)"),
    start_time: str | None = Query(None, description="ISO-8601 lower bound for timestamp"),
    end_time: str | None = Query(None, description="ISO-8601 upper bound for timestamp"),
):
    """Filter observations; all supplied params are combined with AND logic.

    Returns an empty array when nothing matches.
    """
    # ---- parameter validation -------------------------------------------
    if make is not None and make not in VALID_MAKES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown make '{make}'. Known makes: {', '.join(VALID_MAKES)}",
        )
    if model is not None and model not in VALID_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Known models: {', '.join(VALID_MODELS)}",
        )
    if type is not None and type not in VALID_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown type '{type}'. Known types: {', '.join(VALID_TYPES)}",
        )
    if color is not None and color not in VALID_COLORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown color '{color}'. Known colors: {', '.join(VALID_COLORS)}",
        )
    if wheeler is not None and wheeler not in VALID_WHEELERS:
        raise HTTPException(
            status_code=422,
            detail=f"No vehicles with wheeler={wheeler}. Known values: {VALID_WHEELERS}",
        )
    if camera_id is not None and camera_id not in CAMERA_LOC:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown camera_id '{camera_id}'. Known cameras: {', '.join(sorted(CAMERA_LOC))}",
        )
    if speed is not None and min_speed is not None and speed < min_speed:
        raise HTTPException(status_code=422, detail="speed cannot be below min_speed")
    if speed is not None and max_speed is not None and speed > max_speed:
        raise HTTPException(status_code=422, detail="speed cannot be above max_speed")
    if min_speed is not None and max_speed is not None and min_speed > max_speed:
        raise HTTPException(status_code=422, detail="min_speed cannot exceed max_speed")

    # ---- parse + validate time bounds ------------------------------------
    start_dt = _parse_time(start_time, "start_time")
    end_dt = _parse_time(end_time, "end_time")
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise HTTPException(status_code=422, detail="start_time cannot be after end_time")

    # ---- apply all filters (AND) -----------------------------------------
    results = []
    for o in TRAFFIC_DATA:
        if plate is not None and o["plate"]["text"] != plate:
            continue
        if type is not None and o["vehicle"]["type"] != type:
            continue
        if color is not None and o["vehicle"]["color"] != color:
            continue
        if wheeler is not None and o["vehicle"]["wheeler"] != wheeler:
            continue
        if make is not None and o["vehicle"]["make"] != make:
            continue
        if model is not None and o["vehicle"]["model"] != model:
            continue
        if camera_id is not None and o["camera_id"] != camera_id:
            continue

        spd = o["motion"]["speed_kmh"]
        if speed is not None and spd != speed:
            continue
        if min_speed is not None and spd < min_speed:
            continue
        if max_speed is not None and spd > max_speed:
            continue

        ts = datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00"))
        if start_dt is not None and ts < start_dt:
            continue
        if end_dt is not None and ts > end_dt:
            continue

        results.append(o)

    return results


@app.get("/traffic/alert", response_model=AlertResponse)
def get_traffic_alert():
    """Return a randomly generated alert located near the camera network."""
    cam_id, (lat, lon) = random.choice(list(CAMERA_LOC.items()))
    # Nudge the alert a short distance from the chosen camera so it stays
    # within the road corridor instead of sitting exactly on the camera.
    lat += random.uniform(-0.002, 0.002)
    lon += random.uniform(-0.002, 0.002)
    return AlertResponse(
        alert_id=f"ALERT_{random.randint(1000, 9999)}",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        latitude=round(lat, 6),
        longitude=round(lon, 6),
        alert=random.choice(VALID_ALERTS),
        severity=random.choice(VALID_SEVERITIES),
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _parse_time(value: str | None, param: str) -> datetime | None:
    """Parse an ISO-8601 timestamp; 422 with a clear message on bad input."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {param} '{value}'. Use ISO-8601, e.g. 2026-08-27T05:30:00Z",
        )
