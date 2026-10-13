import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pitch
from detect import detect
from events import ball_owners, defended_goals, detect_goals, detect_shots, movement, passes, touches, track_ball
from render import render
from report import write_report
from track import build_people

HEATMAP_BINS = (24, 14)  # 5 m cells
MIN_SHAPE_PLAYERS = 7  # outfield players of a team on screen needed to measure its shape


def heatmap(points):
    xy = np.array(points).reshape(-1, 2)
    grid, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=HEATMAP_BINS, range=[[0, pitch.LENGTH], [0, pitch.WIDTH]])
    return grid.astype(int).tolist()


def build_stats(
    cache, people, ball, owners, spells, pass_events, shot_events, goals, distance, top, team_colours, shots
):
    fps = cache["fps"]
    held = Counter(o for o in owners if o)
    team_held = Counter(o[0] for o in owners if o)
    seen = Counter((p["team"], p["label"]) for plist in people for p in plist if p["kind"] != "referee")

    width, depth, positions = defaultdict(list), defaultdict(list), defaultdict(list)
    for plist in people:
        by_team = defaultdict(list)
        for p in plist:
            if p["pos"] is not None and p["kind"] != "referee":
                positions[p["team"]].append(p["pos"])
                if p["kind"] == "player":
                    by_team[p["team"]].append(p["pos"])
        for k, pts in by_team.items():
            if len(pts) >= MIN_SHAPE_PLAYERS:
                pts = np.array(pts)
                depth[k].append(np.ptp(pts[:, 0]))
                width[k].append(np.ptp(pts[:, 1]))

    def per_team(k):
        done = [e for e in pass_events if e["kind"] == "pass" and e["from"][0] == k]
        lost = [e for e in pass_events if e["kind"] == "turnover" and e["from"][0] == k]
        lengths = [e["length"] for e in done if e["length"] is not None]
        shots_k = [s for s in shot_events if s["team"] == k]
        return {
            "colour_bgr": list(team_colours[k]),
            "possession_frames": team_held[k],
            "passes": len(done),
            "passes_attempted": len(done) + len(lost),
            "avg_pass_length_m": float(np.mean(lengths)) if lengths else None,
            "turnovers_won": sum(1 for e in pass_events if e["kind"] == "turnover" and e["to"][0] == k),
            "shots": len(shots_k),
            "shots_on_target": sum(s["on_target"] for s in shots_k),
            "goals": sum(1 for g in goals if g["team"] == k),
            "width_m": float(np.mean(width[k])) if width[k] else None,
            "depth_m": float(np.mean(depth[k])) if depth[k] else None,
            "heatmap": heatmap(positions[k]),
        }

    players = []
    for (team, label), frames in seen.items():
        if label == "?":
            continue
        key = (team, label)
        players.append(
            {
                "team": team,
                "label": label,
                "seconds": frames / fps,
                "distance_m": distance.get(key, 0.0),
                "top_speed_kmh": top.get(key, 0.0) * 3.6,  # 95th-percentile one-second speed
                "touches": sum(1 for s in spells if s["owner"] == key),
                "passes": sum(1 for e in pass_events if e["kind"] == "pass" and e["from"] == key),
                "possession_s": held[key] / fps,
                "shots": sum(1 for s in shot_events if s["player"] == key),
                "goals": sum(1 for g in goals if g["player"] == key),
            }
        )
    players.sort(key=lambda p: (p["team"], p["label"] != "GK", int(p["label"]) if p["label"].isdigit() else 0))

    per_second = max(int(round(fps)), 1)

    def name(owner):
        return None if owner is None else f"Team {owner[0] + 1} #{owner[1]}"

    return {
        "fps": fps,
        "frames": len(cache["frames"]),
        "analysed_frames": sum(1 for f in cache["frames"] if "dets" in f),
        "pitch_frames": sum(1 for plist in people if any(p["pos"] is not None for p in plist)),
        "ball_frames": sum(1 for b in ball if b and b["detected"]),
        "camera_shots": len(set(shots)),
        "referee_seconds": sum(1 for plist in people if any(p["kind"] == "referee" for p in plist)) / fps,
        "teams": [per_team(k) for k in (0, 1)],
        "players": players,
        "events": sorted(
            [{"frame": g["frame"], "kind": "goal", "team": g["team"], "player": name(g["player"])} for g in goals]
            + [
                {
                    "frame": s["frame"],
                    "kind": "shot on target" if s["on_target"] else "shot",
                    "team": s["team"],
                    "player": name(s["player"]),
                    "speed_kmh": s["speed_kmh"],
                }
                for s in shot_events
                if not s["scored"]
            ],
            key=lambda e: e["frame"],
        ),
        "shots": shot_events,
        "ball_heatmap": heatmap([b["pos"] for b in ball if b and b["detected"] and b["pos"] is not None]),
        "timeline": [
            [sum(1 for o in owners[i : i + per_second] if o and o[0] == k) for k in (0, 1)]
            for i in range(0, len(owners), per_second)
        ],
    }


def main():
    parser = argparse.ArgumentParser(prog="Football Analysis")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--device", default=None, help="e.g. cpu, 0 (first GPU); auto-detected when omitted")
    args = parser.parse_args()

    video = Path(args.video_path)
    out = Path(args.output_dir) / video.stem
    out.mkdir(parents=True, exist_ok=True)

    cache = detect(video, Path(args.output_dir) / "cache" / f"{video.stem}.pkl", args.device)
    fps = cache["fps"]
    print("Tracking players")
    people, team_colours, homographies, shots, calibration_quality = build_people(cache)
    print("Finding ball and events")
    ball = track_ball(cache, people, homographies, shots)
    owners = ball_owners(people, ball)
    spells = touches(owners, shots)
    pass_events = passes(spells, ball)
    direction = defended_goals(people, fps)
    shot_events = detect_shots(spells, ball, shots, fps, direction)
    goals = detect_goals(ball, spells, shot_events, fps, direction)
    distance, top = movement(people, fps)

    print("Rendering annotated video")
    video_out = out / "annotated.mp4"
    render(video, video_out, fps, cache["size"], people, ball, owners, team_colours, goals)

    stats = build_stats(
        cache, people, ball, owners, spells, pass_events, shot_events, goals, distance, top, team_colours, shots
    )
    stats.update(video=str(video), output_video=str(video_out.resolve()), calibration_quality=calibration_quality)
    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=float))

    print(f"Output video: {video_out.resolve()}")
    print(f"Stats: {stats_path.resolve()}")
    print(f"Report: {write_report(stats_path).resolve()}")


if __name__ == "__main__":
    main()
