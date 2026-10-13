"""Pitch geometry in metres and the camera-to-pitch homography from the 32 detected pitch keypoints.

Coordinates follow the pitch model's template (roboflow/sports SoccerPitchConfiguration):
x runs along the 120 m length (goal lines at x=0 and x=120), y across the 70 m width.
"""

import cv2
import numpy as np

LENGTH, WIDTH = 120.0, 70.0
PENALTY_BOX_W, PENALTY_BOX_L = 41.0, 20.15
GOAL_BOX_W, GOAL_BOX_L = 18.32, 5.5
CIRCLE_R, PENALTY_SPOT = 9.15, 11.0
GOAL_W, GOAL_DEPTH = 7.32, 2.5
MIN_KPT_CONF = 0.5
MIN_SPREAD = 50.0  # m², inlier keypoints spanning less than this (e.g. all on one line) give a garbage homography

_L, _W = LENGTH, WIDTH
VERTICES = np.array(
    [
        (0, 0),
        (0, (_W - PENALTY_BOX_W) / 2),
        (0, (_W - GOAL_BOX_W) / 2),
        (0, (_W + GOAL_BOX_W) / 2),
        (0, (_W + PENALTY_BOX_W) / 2),
        (0, _W),
        (GOAL_BOX_L, (_W - GOAL_BOX_W) / 2),
        (GOAL_BOX_L, (_W + GOAL_BOX_W) / 2),
        (PENALTY_SPOT, _W / 2),
        (PENALTY_BOX_L, (_W - PENALTY_BOX_W) / 2),
        (PENALTY_BOX_L, (_W - GOAL_BOX_W) / 2),
        (PENALTY_BOX_L, (_W + GOAL_BOX_W) / 2),
        (PENALTY_BOX_L, (_W + PENALTY_BOX_W) / 2),
        (_L / 2, 0),
        (_L / 2, _W / 2 - CIRCLE_R),
        (_L / 2, _W / 2 + CIRCLE_R),
        (_L / 2, _W),
        (_L - PENALTY_BOX_L, (_W - PENALTY_BOX_W) / 2),
        (_L - PENALTY_BOX_L, (_W - GOAL_BOX_W) / 2),
        (_L - PENALTY_BOX_L, (_W + GOAL_BOX_W) / 2),
        (_L - PENALTY_BOX_L, (_W + PENALTY_BOX_W) / 2),
        (_L - PENALTY_SPOT, _W / 2),
        (_L - GOAL_BOX_L, (_W - GOAL_BOX_W) / 2),
        (_L - GOAL_BOX_L, (_W + GOAL_BOX_W) / 2),
        (_L, 0),
        (_L, (_W - PENALTY_BOX_W) / 2),
        (_L, (_W - GOAL_BOX_W) / 2),
        (_L, (_W + GOAL_BOX_W) / 2),
        (_L, (_W + PENALTY_BOX_W) / 2),
        (_L, _W),
        (_L / 2 - CIRCLE_R, _W / 2),
        (_L / 2 + CIRCLE_R, _W / 2),
    ],
    np.float32,
)
SPOTS = np.array([(PENALTY_SPOT, _W / 2), (_L - PENALTY_SPOT, _W / 2), (_L / 2, _W / 2)], np.float32)


def _box(x0, x1, w):
    return [(x0, (_W - w) / 2), (x1, (_W - w) / 2), (x1, (_W + w) / 2), (x0, (_W + w) / 2)]


# Polylines (metres) for drawing the pitch.
LINES = [
    [(0, 0), (_L, 0), (_L, _W), (0, _W), (0, 0)],
    [(_L / 2, 0), (_L / 2, _W)],
    _box(0, PENALTY_BOX_L, PENALTY_BOX_W),
    _box(_L, _L - PENALTY_BOX_L, PENALTY_BOX_W),
    _box(0, GOAL_BOX_L, GOAL_BOX_W),
    _box(_L, _L - GOAL_BOX_L, GOAL_BOX_W),
    _box(0, -GOAL_DEPTH, GOAL_W),
    _box(_L, _L + GOAL_DEPTH, GOAL_W),
]


def homography(kpts):
    """Image -> pitch (metres) homography from one frame's keypoints, or None if they are too few or degenerate."""
    if kpts is None:
        return None
    ok = kpts[:, 2] > MIN_KPT_CONF
    if ok.sum() < 4:
        return None
    H, inliers = cv2.findHomography(kpts[ok, :2], VERTICES[ok], cv2.RANSAC, 1.0)
    if H is None or inliers.sum() < 4:
        return None
    if cv2.contourArea(cv2.convexHull(VERTICES[ok][inliers.ravel() == 1])) < MIN_SPREAD:
        return None
    return H


def keypoint_agreement(kpts):
    """Share of the confident keypoints that one homography explains within 1 m, or None without a fit.

    Low agreement means the keypoint model mislabelled points (e.g. a penalty-box line read as the halfway line).
    """
    if kpts is None or (kpts[:, 2] > MIN_KPT_CONF).sum() < 4:
        return None
    ok = kpts[:, 2] > MIN_KPT_CONF
    H, inliers = cv2.findHomography(kpts[ok, :2], VERTICES[ok], cv2.RANSAC, 1.0)
    return None if H is None else float(inliers.mean())


def smooth_homographies(raw, shots, size, window=5):
    """Temporal median of each frame's mapping within a camera shot, refitted to a homography.

    Sparse keypoints (e.g. only the halfway line and centre circle) let single-frame fits flip by metres;
    the median over +-window frames votes those flips out and fills short keypoint dropouts.
    """
    w, h = size
    grid = np.array([(x * w, y * h) for x in (0.1, 0.5, 0.9) for y in (0.45, 0.7, 0.95)], np.float32)
    proj = [None if H is None else to_pitch(H, grid) for H in raw]
    out = []
    for i in range(len(raw)):
        near = [
            proj[j]
            for j in range(max(0, i - window), min(len(raw), i + window + 1))
            if proj[j] is not None and shots[j] == shots[i]
        ]
        if len(near) < 3:
            out.append(raw[i])
            continue
        H, _ = cv2.findHomography(grid, np.median(near, axis=0))
        out.append(H)
    return out


def calibration_ids(homographies, shots, size, jump=3.0):
    """Id per frame that changes at every cut, missing homography or sudden jump (> jump metres) in the mapping.

    A camera pans smoothly, so a jump between consecutive frames is the keypoint fit flipping; positions on
    either side of one are not comparable.
    """
    w, h = size
    grid = np.array([(x * w, y * h) for x in (0.25, 0.5, 0.75) for y in (0.6, 0.85)], np.float32)
    ids, current, prev = [], 0, None
    for i, H in enumerate(homographies):
        proj = None if H is None else to_pitch(H, grid)
        if (
            proj is None
            or prev is None
            or shots[i] != shots[i - 1]
            or np.linalg.norm(proj - prev, axis=1).mean() > jump
        ):
            current += 1
        ids.append(current)
        prev = proj
    return ids


def to_pitch(H, points):
    """Map image points (N, 2) to pitch metres (N, 2)."""
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def inside(xy, margin=0.0):
    """True where pitch points lie inside the touch and goal lines (plus margin, metres)."""
    xy = np.asarray(xy).reshape(-1, 2)
    return (xy[:, 0] >= -margin) & (xy[:, 0] <= _L + margin) & (xy[:, 1] >= -margin) & (xy[:, 1] <= _W + margin)


def in_goal(xy):
    """Index of the goal (0 = x=0 end, 1 = x=120 end) a pitch point sits in, or None."""
    x, y = xy
    if abs(y - _W / 2) > GOAL_W / 2 + 0.3:
        return None
    if -GOAL_DEPTH <= x < 0:
        return 0
    if _L < x <= _L + GOAL_DEPTH:
        return 1
    return None
