"""
Mock multi-camera traffic dataset generator.

Simulates vehicles driving through a small road network around
Brahmapur (Berhampur), Odisha, India. A fixed set of cameras is placed
on road intersections/segments; each vehicle is given a route (a list of
cameras) and "travels" along it, producing one observation per camera
whose timestamp is derived from the real distance between cameras and the
vehicle's speed on each segment.

Coordinate model
----------------
The network is a 4x4 grid of road junctions centered on the city
(19.3149 N, 84.7941 E). Every junction is a camera. Roads are straight
segments between junctions, so consecutive detections are always joined by
a straight line -> routes on a map look like continuous polylines.

Timing model
------------
travel_time = segment_distance / speed, so the timestamp difference between
two cameras is consistent with the distance and the vehicle's speed on that
segment. Slow/stopped traffic is simulated by giving some segments a low
speed (and jittering the arrival time slightly).

Identity
--------
Plate text, make, model, color, type and wheeler are constant per vehicle.
The embedding is a deterministic function of the vehicle id + a small
per-observation noise, so observations of the same vehicle stay similar
(cosine distance ~0.02-0.05) while differing enough to look realistic for
re-identification tasks.

Output
------
Writes mock_traffic_data.json (JSON array of observation records).
"""

import json
import math
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

# ---------------------------------------------------------------------------
# Road network: 4x4 grid of junctions around Brahmapur, Odisha.
# Lat grows north, lon grows east. Spacing is ~0.008 deg (~0.9 km), which is
# a realistic urban block size for a city road network.
# ---------------------------------------------------------------------------

BASE_LAT, BASE_LON = 19.3149, 84.7941
SPACING = 0.008  # degrees between adjacent junctions (~0.89 km)

ROWS, COLS = 3, 4  # 3x4 grid of junctions -> 12 cameras

CAMERA_IDS = []
for row in range(ROWS):
    for col in range(COLS):
        cam_id = f"CAM_{row * COLS + col + 1:03d}"
        lat = BASE_LAT + (row - (ROWS - 1) / 2) * SPACING
        lon = BASE_LON + (col - (COLS - 1) / 2) * SPACING
        CAMERA_IDS.append((cam_id, lat, lon))

CAMERA_LOC = {cid: (lat, lon) for cid, lat, lon in CAMERA_IDS}


def camera_at(row: int, col: int) -> str:
    return CAMERA_IDS[row * COLS + col][0]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def camera_distance(cam_a: str, cam_b: str) -> float:
    return haversine_km(*CAMERA_LOC[cam_a], *CAMERA_LOC[cam_b])


# ---------------------------------------------------------------------------
# Vehicle catalog (realistic Indian vehicles).
# Each entry: type, wheeler, make, model, allowed colors.
# ---------------------------------------------------------------------------

VEHICLE_CATALOG = [
    # (type, wheeler, make, model, colors)
    ("car", 4, "Maruti Suzuki", "Swift", ["white", "red", "silver", "blue", "gray"]),
    ("car", 4, "Maruti Suzuki", "Baleno", ["white", "gray", "blue", "silver", "black"]),
    ("car", 4, "Hyundai", "Creta", ["white", "black", "silver", "red", "blue"]),
    ("car", 4, "Hyundai", "i20", ["white", "red", "silver", "gray", "orange"]),
    ("car", 4, "Tata", "Nexon", ["white", "gray", "blue", "silver", "green"]),
    ("car", 4, "Tata", "Punch", ["white", "red", "gray", "silver", "blue"]),
    ("car", 4, "Mahindra", "Scorpio", ["black", "white", "silver", "gray", "red"]),
    ("car", 4, "Toyota", "Innova", ["white", "silver", "black", "gray", "brown"]),
    ("car", 4, "Honda", "City", ["white", "silver", "red", "gray", "blue"]),
    ("car", 4, "Kia", "Seltos", ["white", "gray", "red", "silver", "black"]),
    ("motorcycle", 2, "Royal Enfield", "Classic 350", ["black", "gray", "red", "green"]),
    ("motorcycle", 2, "Hero", "Splendor", ["black", "red", "blue", "gray", "green"]),
    ("motorcycle", 2, "Honda", "Shine", ["black", "red", "blue", "gray", "silver"]),
    ("scooter", 2, "Honda", "Activa", ["white", "red", "gray", "blue", "black"]),
    ("scooter", 2, "TVS", "Jupiter", ["white", "red", "black", "gray", "blue"]),
    ("auto", 3, "Bajaj", "RE Auto", ["green", "yellow", "black"]),
    ("van", 4, "Maruti Suzuki", "Eeco", ["white", "silver", "gray", "blue"]),
    ("van", 4, "Mahindra", "Bolero", ["white", "gray", "black", "silver"]),
    ("bus", 6, "Ashok Leyland", "Viking", ["white", "blue", "red", "orange"]),
    ("bus", 6, "Tata", "Starbus", ["white", "blue", "red", "green"]),
    ("truck", 6, "Ashok Leyland", "Ecomet", ["white", "blue", "red", "gray"]),
    ("truck", 6, "Tata", "Ace", ["white", "blue", "red", "gray", "silver"]),
]

RTO_CODES = ["OD02", "OD05", "OD07", "OD08", "OD10", "OD11", "OD12", "OD14"]

PLATE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I/O/Q to keep plates plausible


def random_plate(rng: random.Random) -> str:
    code = rng.choice(RTO_CODES)
    series = "".join(rng.choice(PLATE_LETTERS) for _ in range(2))
    number = rng.randint(1000, 9999)
    return f"{code}{series}{number}"


# ---------------------------------------------------------------------------
# Embeddings: deterministic per vehicle (seeded), plus per-observation noise.
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 32


def make_base_embedding(seed: int) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [round(v / norm, 4) for v in vec]


def noise_embedding(base: list[float], seed: int) -> list[float]:
    """Base embedding + small noise; keeps cosine similarity ~0.97-0.99."""
    rng = random.Random(seed)
    noisy = [v + rng.gauss(0.0, 0.04) for v in base]
    norm = math.sqrt(sum(v * v for v in noisy))
    return [round(v / norm, 4) for v in noisy]


# ---------------------------------------------------------------------------
# Route generation on the grid.
# A route is a list of junction (row, col) tuples. Consecutive junctions are
# always orthogonally adjacent, so every step is a straight ~0.9 km segment.
# ---------------------------------------------------------------------------

def random_route(rng: random.Random, min_steps: int = 3, max_steps: int = 6) -> list[tuple[int, int]]:
    """Random walk of adjacent grid junctions (no immediate backtracking)."""
    while True:
        row, col = rng.randrange(ROWS), rng.randrange(COLS)
        steps = rng.randint(min_steps, max_steps)
        path = [(row, col)]
        for _ in range(steps):
            options = []
            if row > 0:
                options.append((row - 1, col))
            if row < ROWS - 1:
                options.append((row + 1, col))
            if col > 0:
                options.append((row, col - 1))
            if col < COLS - 1:
                options.append((row, col + 1))
            prev = path[-2] if len(path) >= 2 else None
            options = [o for o in options if o != prev]  # no backtracking
            if not options:
                break
            row, col = rng.choice(options)
            path.append((row, col))
        if len(path) >= 3:  # need at least 3 cameras for a meaningful route
            return path


def segment_speed_kmh(rng: random.Random, vehicle_speed: float) -> float:
    """Per-segment speed; simulates slow/stopped traffic ~15% of the time."""
    if rng.random() < 0.15:
        return round(rng.uniform(5.0, 22.0), 1)  # slow/stopped traffic
    return round(vehicle_speed * rng.uniform(0.85, 1.15), 1)


# ---------------------------------------------------------------------------
# Vehicle simulation.
# ---------------------------------------------------------------------------

def simulate_vehicle(vehicle_idx: int, rng: random.Random) -> list[dict]:
    """Returns observations for one vehicle traveling along a random route."""
    # ---- fixed identity ---------------------------------------------------
    vtype, wheeler, make, model, colors = rng.choice(VEHICLE_CATALOG)
    plate = random_plate(rng)
    color = rng.choice(colors)
    base_embedding = make_base_embedding(1000 + vehicle_idx)
    base_speed = {
        "motorcycle": rng.uniform(32, 55),
        "scooter": rng.uniform(28, 48),
        "car": rng.uniform(32, 62),
        "van": rng.uniform(30, 55),
        "auto": rng.uniform(22, 40),
        "bus": rng.uniform(25, 48),
        "truck": rng.uniform(22, 45),
    }[vtype]

    # ---- route ------------------------------------------------------------
    path = random_route(rng)
    start_minute = rng.randint(0, 12 * 60)  # 00:00 .. 12:00
    t = datetime(2026, 8, 27, tzinfo=timezone.utc) + timedelta(minutes=start_minute)
    t += timedelta(seconds=rng.uniform(0, 50))

    observations: list[dict] = []
    for step, (row, col) in enumerate(path):
        cam_id = camera_at(row, col)

        # geographic position: at the junction, nudged slightly toward the
        # next junction so consecutive points form a smooth line
        lat, lon = CAMERA_LOC[cam_id]
        if step + 1 < len(path):
            nr, nc = path[step + 1]
            lat += (nr - row) * SPACING * 0.18
            lon += (nc - col) * SPACING * 0.18

        # travel time from the previous camera (distance / speed)
        if step > 0:
            prev_cam = camera_at(*path[step - 1])
            dist_km = camera_distance(prev_cam, cam_id)
            seg_speed = segment_speed_kmh(rng, base_speed)
            t += timedelta(seconds=dist_km / seg_speed * 3600 + rng.uniform(-3, 8))

        # direction of travel on the segment (degrees, 0 = east, CCW)
        if step + 1 < len(path):
            nr, nc = path[step + 1]
            dlat, dlon = nr - row, nc - col
            direction = (math.degrees(math.atan2(dlat, dlon)) + 360) % 360
        elif step > 0:
            pr, pc = path[step - 1]
            direction = (math.degrees(math.atan2(row - pr, col - pc)) + 360) % 360
        else:
            direction = 90.0

        speed = segment_speed_kmh(rng, base_speed)

        # realistic image-space bounding box (1280x720 camera frame)
        width = rng.randint(90, 220) if wheeler >= 4 else rng.randint(40, 90)
        height = int(width * rng.uniform(0.8, 1.0))
        x1 = rng.randint(50, 1150 - width)
        y1 = rng.randint(200, 650 - height)

        observations.append({
            "camera_id": cam_id,
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{rng.randint(0, 999):03d}Z",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "plate": {
                "text": plate,
                "confidence": round(rng.uniform(0.82, 0.99), 2),
            },
            "vehicle": {
                "type": vtype,
                "color": color,
                "make": make,
                "model": model,
                "confidence": round(rng.uniform(0.72, 0.93), 2),
                "wheeler": wheeler,
            },
            "motion": {
                "speed_kmh": round(speed, 1),
                "direction": round(direction, 1),
            },
            "tracking": {
                "local_track_id": rng.randint(1, 500),
                "embedding": noise_embedding(base_embedding, vehicle_idx * 100 + step),
            },
            "bbox": [x1, y1, x1 + width, y1 + height],
        })

    return observations


def main():
    rng = random.Random(7)
    n_vehicles = 40
    observations: list[dict] = []

    for vidx in range(n_vehicles):
        observations.extend(simulate_vehicle(vidx, rng))

    # sort by timestamp for a natural reading order
    observations.sort(key=lambda o: o["timestamp"])

    with open("mock_traffic_data.json", "w", encoding="utf-8") as f:
        json.dump(observations, f, indent=2)

    # quick summary
    by_plate: dict[str, list[dict]] = {}
    for o in observations:
        by_plate.setdefault(o["plate"]["text"], []).append(o)

    print(f"vehicles: {n_vehicles}")
    print(f"observations: {len(observations)}")
    print(f"cameras: {len(CAMERA_IDS)}")
    print(f"avg observations per vehicle: {len(observations) / n_vehicles:.1f}")
    plates = sorted(by_plate.keys())
    print("sample plates:", plates[:5], "...")
    print("embedding dim:", EMBEDDING_DIM)


if __name__ == "__main__":
    main()
