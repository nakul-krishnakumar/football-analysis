"""Stage 3: ball trajectory, possession, passes, shots, goals and player movement."""

from collections import defaultdict

import numpy as np

import pitch

BALL_MIN_CONF = 0.15
BALL_REACQUIRE_CONF = 0.4  # confidence needed to pick the ball up with no recent position
BALL_SWITCH_CONF = 0.6  # confidence needed to jump to a ball far from the current trajectory
MAX_BALL_JUMP = 0.08  # fraction of the frame width the ball may move per frame
MAX_BALL_GAP = 12  # missing frames to interpolate over
MAX_BALL_ASPECT = 1.6  # width/height above this is a flat painted spot seen in perspective, not a ball
SPOT_RADIUS, SPOT_PLAYER_RADIUS = 1.0, 4.0  # a "ball" on a painted spot with nobody near it is the spot
PARKED_LINE_DISTANCE, PARKED_CORNER_DISTANCE = 1.5, 2.0  # a "ball" on a line with nobody near is a spare ball
BALL_FIELD_MARGIN = 0.3
CONTROL_RADIUS = 1.5  # metres between feet and ball to control it
CONTROL_RADIUS_BOX = 0.6  # same, as a share of player height, when there is no homography
MIN_HOLD = 3  # frames of control for a touch
MERGE_GAP = 10  # frames a player may lose the ball and still be on the same touch
SHOT_WINDOW = 8  # frames after a touch used to measure the ball's flight (a lofted ball's projection bends later)
SHOT_MIN_SPEED = 12.0  # m/s (43 km/h)
SHOT_MIN_DISTANCE = 2.5  # metres from the goal line; closer starts are scrambles or glitches in the goal mouth
SHOT_MAX_RESIDUAL = 1.5  # metres a ball position may sit off the fitted straight flight
SHOT_MAX_DISTANCE = 30.0  # metres from the goal line
SHOT_WIDE_MARGIN = 8.0  # metres outside the posts still counted as an attempt
GOAL_MIN_FRAMES = 3  # detections inside the goal needed for a goal
GOAL_DEDUPE = 60.0  # seconds; the same goal shown again (replays) within this is counted once
DIRECTION_WINDOW = 30.0  # seconds of goalkeeper positions voted on to know which goal each team defends
MAX_RUN_SPEED = 11.0  # m/s; faster steps are homography glitches
SMOOTH_SECONDS = 0.5  # position smoothing window
DISTANCE_STEP_SECONDS = 0.5  # distance is summed over position samples this far apart
TOP_SPEED_SECONDS = 1.0  # speeds are displacements over this long
SPRINT_PERCENTILE = 95  # "sprint speed" is this percentile of a player's one-second speeds


def parked(pos):
    """Candidates on a touch or goal line away from the corners: spare balls left by the ball boys."""
    edge = np.minimum.reduce([pos[:, 0], pitch.LENGTH - pos[:, 0], pos[:, 1], pitch.WIDTH - pos[:, 1]])
    corners = np.array([(0, 0), (0, pitch.WIDTH), (pitch.LENGTH, 0), (pitch.LENGTH, pitch.WIDTH)], np.float32)
    corner = np.linalg.norm(pos[:, None] - corners[None], axis=2).min(axis=1)
    return (
        (np.abs(edge) < PARKED_LINE_DISTANCE)
        & (corner > PARKED_CORNER_DISTANCE)
        & np.array([pitch.in_goal(p) is None for p in pos], bool)
    )


def track_ball(cache, people, homographies, shots):
    """Per frame: {"xy": image point, "pos": pitch metres or None, "detected": bool} or None."""
    frames, width = cache["frames"], cache["size"][0]
    ball = [None] * len(frames)
    last = last_i = None
    for i, f in enumerate(frames):
        if "dets" not in f:
            continue
        if last_i is not None and shots[last_i] != shots[i]:
            last = None
        cands = np.vstack([f["balls"], f["dets"][f["dets"][:, 5] == 0, :5]])
        w, h = cands[:, 2] - cands[:, 0], cands[:, 3] - cands[:, 1]
        cands = cands[(cands[:, 4] >= BALL_MIN_CONF) & (h > 0) & (w <= MAX_BALL_ASPECT * np.maximum(h, 1e-6))]
        if not len(cands):
            continue
        centres = np.column_stack([(cands[:, 0] + cands[:, 2]) / 2, (cands[:, 1] + cands[:, 3]) / 2])
        H = homographies[i]
        pos = pitch.to_pitch(H, centres) if H is not None else None
        ok = np.ones(len(cands), bool)
        if pos is not None:
            ok &= pitch.inside(pos, BALL_FIELD_MARGIN) | np.array([pitch.in_goal(p) is not None for p in pos])
            feet = np.array([p["pos"] for p in people[i] if p["pos"] is not None]).reshape(-1, 2)
            on_spot = np.linalg.norm(pos[:, None] - pitch.SPOTS[None], axis=2).min(axis=1) < SPOT_RADIUS
            near = (np.linalg.norm(pos[:, None] - feet[None], axis=2) < SPOT_PLAYER_RADIUS).any(axis=1)
            ok &= ~((on_spot | parked(pos)) & ~near)

        chosen = None
        for j in np.argsort(-cands[:, 4]):
            if not ok[j]:
                continue
            if last is not None and i - last_i <= MAX_BALL_GAP:
                if (
                    np.hypot(*(centres[j] - last)) <= MAX_BALL_JUMP * width * (i - last_i)
                    or cands[j, 4] >= BALL_SWITCH_CONF
                ):
                    chosen = j
                    break
            elif cands[j, 4] >= BALL_REACQUIRE_CONF:
                chosen = j
                break
        if chosen is not None:
            ball[i] = {"xy": centres[chosen], "pos": None if pos is None else pos[chosen], "detected": True}
            last, last_i = centres[chosen], i

    known = [i for i, b in enumerate(ball) if b]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= MAX_BALL_GAP and shots[a] == shots[b]:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                pa, pb = ball[a]["pos"], ball[b]["pos"]
                ball[i] = {
                    "xy": ball[a]["xy"] + (ball[b]["xy"] - ball[a]["xy"]) * t,
                    "pos": None if pa is None or pb is None else pa + (pb - pa) * t,
                    "detected": False,
                }
    return ball


def ball_owners(people, ball):
    """Per frame: (team, label) of the player controlling the ball, or None."""
    owners = []
    for plist, b in zip(people, ball):
        best, best_d = None, 1.0
        if b:
            for p in plist:
                if p["kind"] == "referee" or p["label"] == "?":
                    continue
                if b["pos"] is not None and p["pos"] is not None:
                    d = np.hypot(*(b["pos"] - p["pos"])) / CONTROL_RADIUS
                else:
                    x1, y1, x2, y2 = p["box"]
                    d = np.hypot(b["xy"][0] - (x1 + x2) / 2, b["xy"][1] - y2) / (CONTROL_RADIUS_BOX * (y2 - y1))
                if d < best_d:
                    best, best_d = (p["team"], p["label"]), d
        owners.append(best)
    return owners


def touches(owners, shots):
    """Spells of control: [{"owner", "start", "end", "shot"}], at least MIN_HOLD frames, short losses merged."""
    runs, start = [], 0
    for i in range(1, len(owners) + 1):
        if i == len(owners) or owners[i] != owners[start] or shots[i] != shots[start]:
            if owners[start] is not None and i - start >= MIN_HOLD:
                runs.append({"owner": owners[start], "start": start, "end": i - 1, "shot": shots[start]})
            start = i
    merged = []
    for r in runs:
        m = merged[-1] if merged else None
        if m and m["owner"] == r["owner"] and m["shot"] == r["shot"] and r["start"] - m["end"] <= MERGE_GAP:
            m["end"] = r["end"]
        else:
            merged.append(dict(r))
    return merged


def passes(spells, ball):
    """Ball moving between two players in one camera shot: a pass (same team) or a turnover (other team)."""
    events = []
    for a, b in zip(spells, spells[1:]):
        if a["shot"] != b["shot"] or a["owner"] == b["owner"]:
            continue
        pa, pb = ball[a["end"]]["pos"], ball[b["start"]]["pos"]
        events.append(
            {
                "kind": "pass" if a["owner"][0] == b["owner"][0] else "turnover",
                "frame": a["end"],
                "from": a["owner"],
                "to": b["owner"],
                "length": None if pa is None or pb is None else float(np.hypot(*(pb - pa))),
            }
        )
    return events


def defended_goals(people, fps):
    """Per frame: +1 if team k defends goal k, -1 if team k defends goal 1-k, 0 if unknown.

    Voted from where each team's goalkeeper stands over a rolling window, so a half-time switch is followed.
    """
    votes = np.zeros(len(people) + 1)
    for i, plist in enumerate(people):
        for p in plist:
            if p["kind"] == "goalkeeper" and p["pos"] is not None and p["team"] is not None:
                votes[i + 1] += 1 if (p["pos"][0] >= pitch.LENGTH / 2) == bool(p["team"]) else -1
    cum = np.cumsum(votes)
    w = int(DIRECTION_WINDOW * fps)
    idx = np.arange(len(people))
    direction = np.sign(cum[np.minimum(idx + w, len(people))] - cum[np.maximum(idx - w, 0)]).astype(int)
    known = np.flatnonzero(direction)
    if len(known):  # carry the last known direction through stretches with no goalkeeper in view
        direction = direction[known[np.clip(np.searchsorted(known, idx, side="right") - 1, 0, None)]]
    return direction


def attacker(direction, frame, goal):
    """Team attacking `goal` at `frame`, or None when unknown."""
    d = direction[frame] if direction is not None else 0
    return None if d == 0 else (1 - goal if d > 0 else goal)


def straight_flight(t, xy):
    """Inlier mask of the constant-velocity flight most points agree with (one-frame wrong picks are ignored)."""
    best = np.zeros(len(t), bool)
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            v = (xy[b] - xy[a]) / (t[b] - t[a])
            inliers = np.linalg.norm(xy[a] + np.outer(t - t[a], v) - xy, axis=1) <= SHOT_MAX_RESIDUAL
            if inliers.sum() > best.sum():
                best = inliers
    return best


def detect_shots(spells, ball, shots, fps, direction=None):
    """A fast ball heading at a goal from within SHOT_MAX_DISTANCE, credited to the attacking team.

    A touch by a defender just before (a block or deflection) is traced back to the last attacker within 1 s;
    a release by the defending team with no attacker before it is a clearance, not a shot.
    """
    found = []
    for n, s in enumerate(spells):
        e = s["end"]
        pts = [
            (i, ball[i]["pos"])
            for i in range(e, min(e + SHOT_WINDOW, len(ball)))
            if ball[i] and ball[i]["detected"] and ball[i]["pos"] is not None and shots[i] == shots[e]
        ]
        if len(pts) < 4:
            continue
        t = np.array([i for i, _ in pts]) / fps
        xy = np.array([p for _, p in pts])
        keep = straight_flight(t, xy)
        if keep.sum() < 4:
            continue
        t, xy = t[keep], xy[keep]
        vx, vy = np.polyfit(t, xy[:, 0], 1)[0], np.polyfit(t, xy[:, 1], 1)[0]
        speed = float(np.hypot(vx, vy))
        if speed < SHOT_MIN_SPEED or abs(vx) < 1e-6:
            continue
        goal = 0 if vx < 0 else 1
        goal_x = goal * pitch.LENGTH
        start = xy[0]
        if not SHOT_MIN_DISTANCE <= abs(goal_x - start[0]) <= SHOT_MAX_DISTANCE:
            continue
        y_at = start[1] + vy * (goal_x - start[0]) / vx
        wide = abs(y_at - pitch.WIDTH / 2) - pitch.GOAL_W / 2
        if wide > SHOT_WIDE_MARGIN:
            continue
        team = attacker(direction, e, goal)
        shooter = s
        if team is not None and s["owner"][0] != team:
            shooter = next(
                (
                    x
                    for x in reversed(spells[:n])
                    if x["owner"][0] == team and x["shot"] == s["shot"] and e - x["end"] <= fps
                ),
                None,
            )
            if shooter is None:
                continue
        if shooter["owner"][1] == "GK":
            continue  # a goalkeeper's long kick, not a shot
        found.append(
            {
                "frame": shooter["end"],
                "team": shooter["owner"][0],
                "player": shooter["owner"],
                "goal": goal,
                "from": [float(v) for v in start],
                "to": [float(goal_x), float(y_at)],
                "speed_kmh": speed * 3.6,
                "on_target": bool(wide <= 0.2),
                "scored": False,
            }
        )
    return found


def detect_goals(ball, spells, shot_events, fps, direction=None):
    """GOAL_MIN_FRAMES real ball detections inside a goal mouth within 3 s; replays within GOAL_DEDUPE merged."""
    # shortcut: a high shot over the bar can project into the goal mouth; needs ball height (3D) to rule out.
    hits = defaultdict(list)
    for i, b in enumerate(ball):
        if b and b["detected"] and b["pos"] is not None:
            g = pitch.in_goal(b["pos"])
            if g is not None:
                hits[g].append(i)
    goals = []
    for g, frames in hits.items():
        group = [frames[0]]
        for i in frames[1:] + [None]:
            if i is not None and i - group[-1] <= 3 * fps:
                group.append(i)
                continue
            if len(group) >= GOAL_MIN_FRAMES:
                first = group[0]
                shot = next(
                    (s for s in reversed(shot_events) if s["goal"] == g and 0 <= first - s["frame"] <= 5 * fps), None
                )
                spell = next((s for s in reversed(spells) if s["end"] <= first), None)
                team = attacker(direction, first, g)
                if team is None:
                    team = shot["team"] if shot else spell["owner"][0] if spell else None
                if shot and shot["team"] != team:
                    shot = None
                if not any(
                    x["goal"] == g and x["team"] == team and first - x["frame"] < GOAL_DEDUPE * fps for x in goals
                ):
                    goals.append({"frame": first, "goal": g, "team": team, "player": shot["player"] if shot else None})
                    if shot:
                        shot["scored"] = shot["on_target"] = True
            if i is not None:
                group = [i]
    return sorted(goals, key=lambda x: x["frame"])


def movement(people, fps):
    """Per (team, label): distance in metres and sprint speed in m/s, from smoothed pitch positions per track."""
    tracks = defaultdict(list)
    for i, plist in enumerate(people):
        for p in plist:
            if p["pos"] is not None and p["kind"] != "referee" and p["label"] != "?":
                tracks[(p["track"], p["team"], p["label"])].append((i, p["pos"]))
    distance, speeds = defaultdict(float), defaultdict(list)
    for (_, team, label), seq in tracks.items():
        idx = np.array([i for i, _ in seq])
        xy = np.array([p for _, p in seq])
        k = max(3, int(fps * SMOOTH_SECONDS)) | 1
        if len(xy) < k:
            continue
        kernel = np.ones(k) / k  # moving average against homography jitter
        xy = np.column_stack([np.convolve(xy[:, c], kernel, mode="valid") for c in (0, 1)])
        idx = idx[k // 2 : len(idx) - k // 2]
        pos = dict(zip(idx.tolist(), xy))
        step, window = max(1, int(fps * DISTANCE_STEP_SECONDS)), max(1, int(fps * TOP_SPEED_SECONDS))
        for i in range(idx[0], idx[-1] - step + 1, step):
            if i in pos and i + step in pos:
                d = float(np.hypot(*(pos[i + step] - pos[i])))
                if d * fps / step <= MAX_RUN_SPEED:
                    distance[(team, label)] += d
        # Displacement over a whole window cancels jitter that summing small steps would add up.
        for i in idx.tolist():
            if i + window in pos:
                v = float(np.hypot(*(pos[i + window] - pos[i]))) * fps / window
                if v <= MAX_RUN_SPEED:
                    speeds[(team, label)].append(v)
    # A high percentile rather than the maximum: a rare calibration flip sets the max, not the 95th percentile.
    top = {k: float(np.percentile(v, SPRINT_PERCENTILE)) for k, v in speeds.items() if len(v) >= fps}
    return distance, top
