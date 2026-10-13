"""Stage 4: annotated video with labels, ball, possession, score and a minimap."""

import shutil
import subprocess

import cv2
import numpy as np

import pitch
from detect import frames

FONT = cv2.FONT_HERSHEY_SIMPLEX
REFEREE_COLOUR = (0, 0, 0)
UNKNOWN_COLOUR = (128, 128, 128)
MAP_SCALE, MAP_PAD = 2, 6  # minimap pixels per metre, padding


def _minimap_base():
    w, h = int(pitch.LENGTH * MAP_SCALE) + 2 * MAP_PAD, int(pitch.WIDTH * MAP_SCALE) + 2 * MAP_PAD
    img = np.full((h, w, 3), (60, 120, 60), np.uint8)
    for line in pitch.LINES:
        cv2.polylines(img, [_map_pts(line)], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(
        img,
        tuple(_map_pts([(pitch.LENGTH / 2, pitch.WIDTH / 2)])[0]),
        int(pitch.CIRCLE_R * MAP_SCALE),
        (255, 255, 255),
        1,
    )
    return img


def _map_pts(points):
    return (np.asarray(points, np.float32) * MAP_SCALE + MAP_PAD).astype(np.int32)


def _colour(p, team_colours):
    if p["kind"] == "referee":
        return REFEREE_COLOUR
    return team_colours[p["team"]] if p["label"] != "?" else UNKNOWN_COLOUR


def _panel(frame, lines):
    h = 12 + 26 * len(lines)
    cv2.rectangle(frame, (10, 10), (330, 10 + h), (40, 40, 40), -1)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (20, 36 + 26 * k), FONT, 0.6, colour, 2, cv2.LINE_AA)


def _to_h264(raw, output_path):
    """Browsers can't play OpenCV's mp4v; re-encode with ffmpeg when it is installed, else keep mp4v."""
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    if shutil.which("ffmpeg") and subprocess.run(cmd).returncode == 0:
        raw.unlink()
    else:
        raw.replace(output_path)


def render(video_path, output_path, fps, size, people, ball, owners, team_colours, goals):
    raw = output_path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {raw}")
    base_map = _minimap_base()
    goal_frames = {g["frame"]: g["team"] for g in goals}
    possession, score = [0, 0], [0, 0]
    try:
        for i, frame in enumerate(frames(video_path)):
            if i >= len(people):
                break
            if goal_frames.get(i) is not None:
                score[goal_frames[i]] += 1
            owner = owners[i]
            if owner:
                possession[owner[0]] += 1

            minimap = base_map.copy() if any(p["pos"] is not None for p in people[i]) else None
            for p in people[i]:
                x1, y1, x2, y2 = (int(v) for v in p["box"])
                colour = _colour(p, team_colours)
                cx = (x1 + x2) // 2
                cv2.ellipse(
                    frame,
                    (cx, y2),
                    (max((x2 - x1) // 2, 1), max((x2 - x1) // 6, 1)),
                    0,
                    -45,
                    235,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(frame, p["label"], (cx - 8, y2 + 20), FONT, 0.55, colour, 2, cv2.LINE_AA)
                if owner and p["kind"] != "referee" and (p["team"], p["label"]) == owner:
                    cv2.fillPoly(frame, [np.array([[cx, y1 - 5], [cx - 9, y1 - 20], [cx + 9, y1 - 20]])], (0, 0, 255))
                if minimap is not None and p["pos"] is not None:
                    cv2.circle(minimap, tuple(_map_pts([p["pos"]])[0]), 4, colour, -1, cv2.LINE_AA)
            b = ball[i]
            if b:
                bx, by = (int(v) for v in b["xy"])
                cv2.fillPoly(frame, [np.array([[bx, by - 8], [bx - 8, by - 22], [bx + 8, by - 22]])], (0, 255, 0))
                if minimap is not None and b["pos"] is not None:
                    cv2.circle(minimap, tuple(_map_pts([b["pos"]])[0]), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(minimap, tuple(_map_pts([b["pos"]])[0]), 3, (0, 0, 0), 1, cv2.LINE_AA)

            total = sum(possession)
            lines = [(f"Score  {score[0]} - {score[1]}", (255, 255, 255))]
            if total:
                lines += [
                    (f"Team {k + 1} possession {100 * possession[k] / total:.0f}%", team_colours[k]) for k in (0, 1)
                ]
            _panel(frame, lines)
            if minimap is not None:
                mh, mw = minimap.shape[:2]
                y0, x0 = frame.shape[0] - mh - 10, 10
                frame[y0 : y0 + mh, x0 : x0 + mw] = cv2.addWeighted(
                    frame[y0 : y0 + mh, x0 : x0 + mw], 0.25, minimap, 0.75, 0
                )
            writer.write(frame)
    finally:
        writer.release()
    _to_h264(raw, output_path)
