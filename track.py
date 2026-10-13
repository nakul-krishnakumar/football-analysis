"""Stage 2: turn cached detections into people with a role, a team and a stable label per frame.

Rules from the game itself: one referee on the pitch, one goalkeeper per end, and at most 10 outfield players
per team, who keep labels 1-10. When the tracker loses a player, the new track takes the free label whose last
pitch position is nearest (appearance re-ID is unreliable from a high broadcast camera).
"""

from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils.matching import linear_assignment
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

import pitch

GOALKEEPER, PLAYER, REFEREE = 1, 2, 3  # player-model class ids (0 is ball)
MIN_CONF = 0.3
CUT_CORRELATION = 0.6  # colour-histogram correlation between frames below this = camera cut
FIELD_MARGIN = 0.5  # metres outside the lines a person's feet may be (line width, homography error)
SLOT_TIMEOUT = 30  # frames a lost player keeps their label before another track may take it
TEAM_SIZE = 10  # outfield labels per team
OUTLIER = 0.5  # kit further from both teams than this share of the gap between the two team colours = GK/referee
TOUCHLINE_BAND = 2.5  # metres; an official whose median position is this close to a touchline is an assistant
GOOD_AGREEMENT = 0.8  # a frame's keypoint fit is trusted when it explains this share of its keypoints
GK_ZONE = 25.0  # metres from a goal line where a kit outlier with no class votes is taken for a goalkeeper


def find_shots(frames):
    """Camera-shot index per frame; a new shot starts at every cut or pitch/non-pitch change."""
    shots, shot = [], 0
    for i, f in enumerate(frames):
        if i and (
            ("dets" in f) != ("dets" in frames[i - 1])
            or cv2.compareHist(f["hist"], frames[i - 1]["hist"], cv2.HISTCMP_CORREL) < CUT_CORRELATION
        ):
            shot += 1
        shots.append(shot)
    return shots


def to_lab(bgr):
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8)[:, None], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)


def fit_teams(colours):
    """Two jersey-colour centres (Lab) from outfield-player samples, refitted once without outliers."""
    lab = to_lab(colours)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    for _ in range(2):
        _, labels, centres = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        dist = np.linalg.norm(lab - centres[labels.ravel()], axis=1)
        keep = dist < 2.5 * np.median(dist) + 1
        lab, colours = lab[keep], colours[keep]
    labels = labels.ravel()[keep]
    display = [tuple(int(v) for v in colours[labels == k].mean(axis=0)) for k in (0, 1)]  # BGR for drawing
    return centres, display


def _tracker():
    return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


def _feet(det):
    return ((det[0] + det[2]) / 2, det[3])


def build_people(cache):
    """Per frame: list of people dicts {kind, team, label, box, pos (metres or None), track}; also team colours.

    kind is "player", "goalkeeper" or "referee"; team is 0/1 (None for the referee); label is "1".."10",
    "GK" or "REF" ("?" for a player track that found no free label).
    """
    frames = cache["frames"]
    width, height = cache["size"]
    shots = find_shots(frames)
    homographies = pitch.smooth_homographies([pitch.homography(f.get("kpts")) for f in frames], shots, cache["size"])
    calibration = pitch.calibration_ids(homographies, shots, cache["size"])
    agreement = [a for a in (pitch.keypoint_agreement(f.get("kpts")) for f in frames) if a is not None]
    calibration_quality = float(np.mean([a >= GOOD_AGREEMENT for a in agreement])) if agreement else 0.0

    # 1. Track everyone inside the touch and goal lines.
    observations = defaultdict(list)  # track key -> [(frame, det, colour, pos)]
    tracker, current_shot = None, None
    for i, f in enumerate(frames):
        if "dets" not in f:
            continue
        if shots[i] != current_shot:
            tracker, current_shot = _tracker(), shots[i]
        H = homographies[i]
        dets, colours = f["dets"], f["colours"]
        keep = (dets[:, 4] >= MIN_CONF) & (dets[:, 5] != 0)
        pos = pitch.to_pitch(H, [_feet(d) for d in dets]) if H is not None and len(dets) else None
        if pos is not None:
            keep &= pitch.inside(pos, FIELD_MARGIN)
        idx = np.flatnonzero(keep)
        tracks = tracker.update(Boxes(dets[idx], (height, width)).cpu().numpy()) if len(idx) else []
        for t in tracks:
            j = idx[int(t[7])]
            observations[(current_shot, int(t[4]))].append((i, dets[j], colours[j], None if pos is None else pos[j]))

    # 2. Team colours from outfield players.
    samples = np.array(
        [c for obs in observations.values() for _, d, c, _ in obs if d[5] == PLAYER and d[4] > 0.5], np.float32
    ).reshape(-1, 3)
    samples = samples[~np.isnan(samples).any(axis=1)]
    if len(samples) < 10:
        raise RuntimeError("Too few players detected to tell the teams apart")
    centres, team_colours = fit_teams(samples)

    # 3. Role per track from the model's class votes, overruled by kit colour: a kit far from both teams is
    # a goalkeeper or the referee even when the model says "player".
    info = {}
    for key, obs in observations.items():
        votes = np.zeros(4)
        for _, d, _, _ in obs:
            votes[int(d[5])] += d[4]
        cols = np.array([c for _, _, c, _ in obs])
        cols = cols[~np.isnan(cols).any(axis=1)]
        lab = to_lab(np.median(cols, axis=0)[None])[0] if len(cols) else None
        dist = None if lab is None else float(np.linalg.norm(centres - lab, axis=1).min())
        xy = [p for _, _, _, p in obs if p is not None]
        info[key] = (votes, dist, np.median(xy, axis=0) if xy else None, cols)
    outlier = OUTLIER * float(np.linalg.norm(centres[0] - centres[1]))

    role, team = {}, {}
    for key, (votes, dist, where, cols) in info.items():
        if votes.argmax() == PLAYER and not (dist is not None and dist > outlier):
            r = "player"
        elif votes[GOALKEEPER] != votes[REFEREE]:
            r = "goalkeeper" if votes[GOALKEEPER] > votes[REFEREE] else "referee"
        elif where is not None and min(where[0], pitch.LENGTH - where[0]) < GK_ZONE:
            r = "goalkeeper"
        else:
            r = "referee"
        if r == "referee" and where is not None and min(where[1], pitch.WIDTH - where[1]) < TOUCHLINE_BAND:
            continue  # assistant referee running the touchline
        role[key] = r
        if r == "player" and len(cols):
            votes = np.linalg.norm(to_lab(cols)[:, None] - centres[None], axis=2).argmin(axis=1)
            team[key] = int(np.bincount(votes, minlength=2).argmax())

    by_frame = defaultdict(list)
    for key, obs in observations.items():
        if key in role:
            for i, d, _, p in obs:
                by_frame[i].append((key, d, p))

    # A goalkeeper belongs to the team whose outfield players stand closer to them on average.
    gk_votes = defaultdict(Counter)
    for plist in by_frame.values():
        xs = defaultdict(list)
        for key, d, p in plist:
            if key in team:
                xs[team[key]].append(p[0] if p is not None else _feet(d)[0])
        if len(xs) < 2:
            continue
        for key, d, p in plist:
            if role[key] == "goalkeeper":
                x = p[0] if p is not None else _feet(d)[0]
                gk_votes[key][min(xs, key=lambda k: abs(np.mean(xs[k]) - x))] += 1
    team.update(goalkeeper_teams(gk_votes, {k: info[k][3] for k in gk_votes}, outlier))

    # 4. Labels 1-10 per team, then one referee per frame and one goalkeeper per end.
    labels = assign_labels(by_frame, team, role)
    people = [[] for _ in frames]
    for i, plist in by_frame.items():
        refs = [x for x in plist if role[x[0]] == "referee"]
        ends = defaultdict(list)
        for x in plist:
            if role[x[0]] == "goalkeeper" and x[0] in team:
                key, d, p = x
                ends[p[0] < pitch.LENGTH / 2 if p is not None else _feet(d)[0] < width / 2].append(x)
        best = [max(refs, key=lambda x: x[1][4])] if refs else []
        best += [max(xs, key=lambda x: x[1][4]) for xs in ends.values()]
        for key, d, p in [x for x in plist if role[x[0]] == "player" and x[0] in team] + best:
            kind = role[key]
            label = {"goalkeeper": "GK", "referee": "REF"}.get(kind) or labels.get((i, key), "?")
            people[i].append(
                {
                    "kind": kind,
                    "team": team.get(key),
                    "label": label,
                    "box": d[:4],
                    "pos": p,
                    "track": (key, calibration[i]),
                }
            )
    return people, team_colours, homographies, shots, calibration_quality


def goalkeeper_teams(votes, colours, min_gap):
    """Team per goalkeeper track: tracks are grouped into the two keeper kits, and each kit takes one team.

    Single-track votes (which team stands nearer) are noisy, so they are pooled per kit; if both kits pick the
    same team, the less certain one takes the other.
    """
    keys = [k for k in votes if len(colours[k])]
    if not keys:
        return {}
    lab = np.array([to_lab(np.median(colours[k], axis=0)[None])[0] for k in keys], np.float32)
    kit = np.zeros(len(keys), int)
    if len(keys) >= 2:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centres = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        if np.linalg.norm(centres[0] - centres[1]) >= min_gap:
            kit = labels.ravel()
    pooled = [sum((votes[k] for k, c in zip(keys, kit) if c == n), Counter()) for n in (0, 1)]
    choice = [p.most_common(1)[0][0] if p else None for p in pooled]
    if choice[0] is not None and choice[0] == choice[1]:
        share = [pooled[n][choice[n]] / sum(pooled[n].values()) for n in (0, 1)]
        weaker = int(np.argmin(share))
        choice[weaker] = 1 - choice[weaker]
    return {k: choice[c] for k, c in zip(keys, kit) if choice[c] is not None}


def assign_labels(by_frame, team, role):
    """Map (frame, track key) -> "1".."10": tracks keep their label; new tracks take the nearest free one."""
    slots = {k: {s: {"key": None, "last": -(10**9), "pos": None} for s in range(1, TEAM_SIZE + 1)} for k in (0, 1)}
    held = {}  # track key -> slot
    out = {}
    for i in sorted(by_frame):
        for k in (0, 1):
            visible = [(key, p) for key, _, p in by_frame[i] if team.get(key) == k and role[key] == "player"]
            new = []
            for key, p in visible:
                s = held.get(key)
                if s is not None and slots[k][s]["key"] == key:
                    slots[k][s].update(last=i, pos=p if p is not None else slots[k][s]["pos"])
                    out[(i, key)] = str(s)
                else:
                    new.append((key, p))
            if not new:
                continue
            taken = {held[key] for key, _ in visible if (i, key) in out}
            shot = new[0][0][0]  # tracks from an earlier camera shot never come back
            free = [
                s
                for s, st in slots[k].items()
                if s not in taken and (st["key"] is None or st["key"][0] != shot or i - st["last"] > SLOT_TIMEOUT)
            ]
            if not free:
                free = [s for s, st in slots[k].items() if s not in taken and st["last"] < i]
            if not free:
                continue
            cost = np.array(
                [
                    [
                        np.hypot(*(p - slots[k][s]["pos"]))
                        if p is not None and slots[k][s]["pos"] is not None
                        else 60.0
                        for s in free
                    ]
                    for _, p in new
                ]
            )
            cost += np.array(free)[None] * 1e-3  # ties go to the lowest free label
            matches, _, _ = linear_assignment(cost, thresh=1e9)
            for r, c in matches:
                key, p = new[r]
                s = free[c]
                slots[k][s].update(key=key, last=i, pos=p)
                held[key] = s
                out[(i, key)] = str(s)
    return out
