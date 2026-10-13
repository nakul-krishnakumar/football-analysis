"""Stage 1: run the models over every frame and cache the raw outputs, so later stages never re-run inference.

The cache is checkpointed, so an interrupted run resumes where it stopped.
"""

import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PLAYER_MODEL = "models/football-player-detection.pt"  # classes: 0 ball, 1 goalkeeper, 2 player, 3 referee
BALL_MODEL = "models/football-ball-detection.pt"
PITCH_MODEL = "models/football-pitch-detection.pt"  # 32 pitch keypoints, see pitch.VERTICES
MIN_GREEN = 0.25  # share of grass pixels below which a frame is a crowd/close-up shot and models are skipped
CHECKPOINT_EVERY = 250


def frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            yield frame
    finally:
        cap.release()


def green_mask(frame):
    return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (35, 40, 40), (85, 255, 255))


def colour_hist(frame):
    """Small hue/saturation histogram, compared between frames to find camera cuts."""
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).ravel()


def jersey_colour(frame, green, box):
    """Mean BGR of the shirt area, ignoring grass pixels; NaN when too little shirt is visible."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    rows, cols = slice(max(y1 + h // 6, 0), max(y1 + h // 2, 0)), slice(max(x1 + w // 4, 0), max(x2 - w // 4, 0))
    keep = green[rows, cols] == 0
    if keep.sum() < 10:
        return np.full(3, np.nan, np.float32)
    return frame[rows, cols][keep].mean(axis=0).astype(np.float32)


def _model_key():
    missing = [p for p in (PLAYER_MODEL, BALL_MODEL, PITCH_MODEL) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model weights {missing}; run ./download_models.sh")
    return [(p, os.path.getsize(p)) for p in (PLAYER_MODEL, BALL_MODEL, PITCH_MODEL)]


def _save(cache, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.replace(path)  # atomic, so a crash mid-write never corrupts the checkpoint


def detect(video_path, cache_path, device=None):
    """Return {"fps", "size", "frames": [per-frame dict]}; each analysed frame has dets, balls, kpts, colours."""
    video_path, cache_path = Path(video_path), Path(cache_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    key = {"video": str(video_path.resolve()), "mtime": video_path.stat().st_mtime, "models": _model_key()}
    cache = None
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("key") != key:
            print("Detection cache is for another video or model version, starting over")
            cache = None
    if cache is None:
        cache = {"key": key, "fps": fps, "size": size, "frames": [], "complete": False}
    if cache["complete"]:
        print(f"Using cached detections: {cache_path}")
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    players, balls, pitch = YOLO(PLAYER_MODEL), YOLO(BALL_MODEL), YOLO(PITCH_MODEL)
    done = len(cache["frames"])
    if done:
        print(f"Resuming detection at frame {done}")
    start = time.time()
    for i, frame in enumerate(frames(video_path)):
        if i < done:
            continue
        green = green_mask(frame)
        record = {"green": green.mean() / 255, "hist": colour_hist(frame)}
        if record["green"] >= MIN_GREEN:
            dets = players.predict(frame, imgsz=1280, conf=0.1, device=device, verbose=False)[0].boxes.data
            ball = balls.predict(frame, imgsz=1280, conf=0.05, device=device, verbose=False)[0].boxes.data
            kpts = pitch.predict(frame, device=device, verbose=False)[0].keypoints
            dets = dets.cpu().numpy()
            record.update(
                dets=dets,  # x1, y1, x2, y2, conf, cls
                balls=ball.cpu().numpy()[:, :5],  # x1, y1, x2, y2, conf
                kpts=kpts.data[0].cpu().numpy() if kpts is not None and len(kpts) else None,  # 32 x (x, y, conf)
                colours=np.array([jersey_colour(frame, green, d[:4]) for d in dets], np.float32).reshape(-1, 3),
            )
        cache["frames"].append(record)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save(cache, cache_path)
            rate = (i + 1 - done) / (time.time() - start)
            print(f"Detected {i + 1}/{total} frames, {rate:.1f} fps, ~{(total - i - 1) / rate / 60:.0f} min left")
    cache["complete"] = True
    _save(cache, cache_path)
    return cache
