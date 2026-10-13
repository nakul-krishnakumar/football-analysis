import cv2
import numpy as np

import pitch
from events import detect_goals, detect_shots, passes, touches, track_ball
from track import TEAM_SIZE, assign_labels

FPS = 25


def camera_h():
    """A plausible pitch -> image homography (left half of the pitch seen from the stand)."""
    src = np.float32([(0, 0), (60, 0), (60, 70), (0, 70)])
    dst = np.float32([(100, 100), (1180, 100), (1280, 700), (0, 700)])
    return cv2.getPerspectiveTransform(src, dst)


def keypoints(H_pitch_to_img, visible):
    img = pitch.to_pitch(H_pitch_to_img, pitch.VERTICES)  # same transform, used in the other direction
    k = np.zeros((32, 3), np.float32)
    k[:, :2] = img
    k[visible, 2] = 0.9
    return k


def test_homography_round_trip():
    Hc = camera_h()
    H = pitch.homography(keypoints(Hc, [0, 1, 5, 9, 12, 13, 16]))
    assert H is not None
    xy = pitch.to_pitch(H, pitch.to_pitch(Hc, [(30.0, 20.0)]))[0]
    assert np.allclose(xy, (30, 20), atol=0.01)


def test_collinear_keypoints_rejected():
    assert pitch.homography(keypoints(camera_h(), [13, 14, 15, 16])) is None  # all on the halfway line


def test_field_and_goal_zones():
    assert pitch.inside([(0.0, 0.0), (120, 70)]).all()
    assert not pitch.inside([(-1.0, 35)]).any()
    assert pitch.in_goal((-1.0, 35)) == 0 and pitch.in_goal((121, 33)) == 1
    assert pitch.in_goal((-1.0, 10)) is None and pitch.in_goal((50, 35)) is None


def frame_with(H_img_to_pitch_inv, balls, players=()):
    dets = np.array([[*b, 0.9, 2] for b in players], np.float32).reshape(-1, 6)
    return {
        "dets": dets,
        "balls": np.array(balls, np.float32).reshape(-1, 5),
        "kpts": None,
        "colours": np.zeros((len(dets), 3), np.float32),
        "green": 0.8,
        "hist": np.zeros(256, np.float32),
    }


def test_ball_filters_spot_outside_and_flat_blobs():
    Hc = camera_h()
    H = np.linalg.inv(Hc)  # image -> pitch

    def box_at(xy, w=10, h=10, conf=0.9):
        x, y = pitch.to_pitch(Hc, [xy])[0]
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf]

    cache = {
        "size": (1280, 720),
        "frames": [
            frame_with(None, [box_at(pitch.SPOTS[0])]),  # penalty spot, nobody near
            frame_with(None, [box_at((-1.5, 10))]),  # spare ball behind the goal line, not in the goal
            frame_with(None, [box_at((30, 30), w=30, h=8)]),  # flat painted mark
            frame_with(None, [box_at((119.5, 15))]),  # spare ball parked on the goal line
            frame_with(None, [box_at((30, 30))]),  # real ball
        ],
    }
    people = [[] for _ in cache["frames"]]
    ball = track_ball(cache, people, [H] * 5, [0] * 5)
    assert ball[:4] == [None, None, None, None]
    assert np.allclose(ball[4]["pos"], (30, 30), atol=0.1)


def test_ball_on_spot_kept_for_penalty_taker():
    Hc = camera_h()
    x, y = pitch.to_pitch(Hc, [pitch.SPOTS[0]])[0]
    cache = {"size": (1280, 720), "frames": [frame_with(None, [[x - 5, y - 5, x + 5, y + 5, 0.9]])]}
    people = [[{"kind": "player", "team": 0, "label": "9", "box": [0, 0, 1, 1], "pos": pitch.SPOTS[0] + (2, 0)}]]
    assert track_ball(cache, people, [np.linalg.inv(Hc)], [0])[0] is not None


def test_labels_reused_within_ten_and_nearest_free_slot():
    team = {(0, t): 0 for t in range(1, 13)} | {(1, t): 0 for t in range(1, 3)}
    role = dict.fromkeys(team, "player")
    by_frame = {0: [((0, t), None, np.array([10.0 * t, 30.0])) for t in range(1, 13)]}  # 12 tracks visible
    first = assign_labels(by_frame, team, role)
    labelled = [t for t in range(1, 13) if (0, (0, t)) in first]
    assert sorted(int(first[(0, (0, t))]) for t in labelled) == list(range(1, TEAM_SIZE + 1))  # 2 left as "?"

    # After a cut, two new tracks appear next to where two labelled players were last seen.
    a, b = labelled[2], labelled[6]
    by_frame[1] = [((1, 1), None, np.array([10.0 * a + 0.5, 30.0])), ((1, 2), None, np.array([10.0 * b - 0.5, 30.0]))]
    labels = assign_labels(by_frame, team, role)
    assert labels[(1, (1, 1))] == first[(0, (0, a))] and labels[(1, (1, 2))] == first[(0, (0, b))]


def test_passes_and_turnovers():
    a, b, c = (0, "1"), (0, "2"), (1, "5")
    owners = [a] * 4 + [None] * 3 + [b] * 4 + [c] * 2 + [b] * 2 + [c] * 4
    shots = [0] * len(owners)
    ball = [{"pos": np.array([float(i), 0.0])} for i in range(len(owners))]
    spells = touches(owners, shots)
    assert [s["owner"] for s in spells] == [a, b, c]  # the 2-frame blip by c is not a touch
    events = passes(spells, ball)
    assert [e["kind"] for e in events] == ["pass", "turnover"]
    assert events[0]["length"] == 4.0

    assert passes(touches([a] * 4 + [b] * 4, [0] * 4 + [1] * 4), ball) == []  # no pass across a camera cut


def ball_path(points):
    return [{"pos": np.array(p, float), "xy": np.zeros(2), "detected": True} for p in points]


def test_shot_on_target_and_slow_pass_ignored():
    spell = [{"owner": (1, "9"), "start": 0, "end": 0, "shot": 0}]
    fast = ball_path([(100 + 1.0 * i, 35 + 0.05 * i) for i in range(10)])  # 25 m/s at the x=120 goal
    shots = detect_shots(spell, fast, [0] * 10, FPS)
    assert len(shots) == 1 and shots[0]["goal"] == 1 and shots[0]["on_target"]
    slow = ball_path([(100 + 0.2 * i, 35) for i in range(10)])  # 5 m/s
    assert detect_shots(spell, slow, [0] * 10, FPS) == []


def test_goal_counted_once_across_replay():
    n = 40 * FPS
    ball = [None] * n
    for start in (100, 100 + 20 * FPS):  # the goal, then its replay 20 s later
        for i in range(start, start + 5):
            ball[i] = ball_path([(121.0, 35.0)])[0]
    spells = [{"owner": (0, "9"), "start": 90, "end": 95, "shot": 0}]
    goals = detect_goals(ball, spells, [], FPS)
    assert len(goals) == 1 and goals[0]["team"] == 0 and goals[0]["goal"] == 1


def test_attacking_team_from_goalkeeper_ends():
    from events import attacker, defended_goals

    gk = {"kind": "goalkeeper", "team": 1, "label": "GK", "box": [0, 0, 1, 1], "pos": np.array([115.0, 35.0])}
    direction = defended_goals([[gk]] * 10, FPS)  # team 1 keeper stands at the x=120 goal
    assert attacker(direction, 5, 1) == 0 and attacker(direction, 5, 0) == 1


def test_clearance_is_not_a_shot_and_block_credits_attacker():
    fast = ball_path([(100 + 1.0 * i, 35) for i in range(10)])  # towards the x=120 goal
    team0_attacks_goal1 = np.ones(10, int)  # team k defends goal k
    clearance = [{"owner": (1, "GK"), "start": 0, "end": 0, "shot": 0}]
    assert detect_shots(clearance, fast, [0] * 10, FPS, team0_attacks_goal1) == []
    blocked = [
        {"owner": (0, "9"), "start": 0, "end": 0, "shot": 0},
        {"owner": (1, "4"), "start": 0, "end": 0, "shot": 0},
    ]
    shots = detect_shots(blocked, fast, [0] * 10, FPS, team0_attacks_goal1)
    assert shots and all(s["player"] == (0, "9") for s in shots)


def test_shot_survives_a_wrong_ball_pick():
    path = [(100 + 1.0 * i, 35) for i in range(10)]
    path[3] = (80.0, 60.0)  # one frame where another object was taken for the ball
    spell = [{"owner": (1, "9"), "start": 0, "end": 0, "shot": 0}]
    assert len(detect_shots(spell, ball_path(path), [0] * 10, FPS)) == 1
