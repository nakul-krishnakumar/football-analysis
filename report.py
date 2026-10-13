"""Build report.html next to a stats.json written by main.py. Usage: python report.py output/<video>/stats.json"""

import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pitch  # noqa: E402

MIN_CALIBRATION_QUALITY = 0.5  # below this share of well-calibrated frames, distance and speed are not shown
MAX_SHOWN_SPEED = 140  # km/h; above this the measured shot speed is inflated by the ball being in the air


def hex_colour(bgr):
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _pitch_axes(title):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.set_facecolor("#3d7a3d")
    for line in pitch.LINES:
        xy = np.array(line)
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=1)
    ax.add_patch(plt.Circle((pitch.LENGTH / 2, pitch.WIDTH / 2), pitch.CIRCLE_R, fill=False, color="white", lw=1))
    ax.set(xlim=(-4, pitch.LENGTH + 4), ylim=(pitch.WIDTH + 2, -2), aspect="equal", title=title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def heatmap_chart(grid, title, colour):
    grid = np.array(grid, float)
    fig, ax = _pitch_axes(title)
    if grid.sum():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("h", [(1, 1, 1, 0), colour])
        ax.imshow(grid.T / grid.max(), extent=(0, pitch.LENGTH, pitch.WIDTH, 0), cmap=cmap, alpha=0.85, zorder=0.5)
    return _png(fig)


def shot_map(stats, colours):
    fig, ax = _pitch_axes("Shots (star = goal, filled = on target)")
    for s in stats["shots"]:
        c = colours[s["team"]]
        (x0, y0), (x1, y1) = s["from"], s["to"]
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "->", "color": c, "lw": 1.2, "alpha": 0.8})
        marker = "*" if s["scored"] else "o"
        ax.scatter(
            [x0],
            [y0],
            s=140 if s["scored"] else 45,
            marker=marker,
            color=c if s["on_target"] else "none",
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )
    return _png(fig)


def possession_chart(stats, colours):
    counts = np.array(stats["timeline"], dtype=float).reshape(-1, 2)
    cumulative = counts.cumsum(axis=0)
    total = cumulative.sum(axis=1)
    seconds = np.arange(len(counts))[total > 0]
    if not len(seconds):
        return "<p class=muted>No possession recorded.</p>"
    share = cumulative[total > 0, 0] / total[total > 0] * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.stackplot(
        seconds, share, 100 - share, colors=colours, edgecolor="#333", linewidth=0.5, labels=["Team 1", "Team 2"]
    )
    ax.set(xlabel="Time (s)", ylabel="Cumulative possession %", ylim=(0, 100), xlim=(0, len(counts)))
    ax.legend(loc="upper right")
    return _png(fig)


def _fmt(v, unit="", digits=0):
    return "–" if v is None else f"{v:.{digits}f}{unit}"


def _speed(event):
    if "speed_kmh" not in event:
        return ""
    if event["speed_kmh"] > MAX_SHOWN_SPEED:
        return " · lofted"  # ground-plane speed of a ball in the air is meaningless
    return f" · {event['speed_kmh']:.0f} km/h"


def _clock(frame, fps):
    s = int(frame / fps)
    return f"{s // 60}:{s % 60:02d}"


def write_report(stats_path):
    stats_path = Path(stats_path)
    stats = json.loads(stats_path.read_text())
    fps, teams = stats["fps"], stats["teams"]
    colours = [hex_colour(t["colour_bgr"]) for t in teams]
    held = sum(t["possession_frames"] for t in teams) or 1
    frames = stats["frames"] or 1
    analysed = stats["analysed_frames"] or 1

    cards = [
        ("Duration", f"{stats['frames'] / fps:.0f} s"),
        ("Pitch in view", f"{100 * stats['analysed_frames'] / frames:.0f}% of frames"),
        ("Pitch calibrated", f"{100 * stats['pitch_frames'] / analysed:.0f}% of pitch frames"),
        ("Ball detected", f"{100 * stats['ball_frames'] / analysed:.0f}% of pitch frames"),
        ("Camera shots", str(stats["camera_shots"])),
    ]

    def row(name, values):
        return f"<tr><th>{name}</th>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"

    team_table = "".join(
        [
            row("Possession", [f"{100 * t['possession_frames'] / held:.0f}%" for t in teams]),
            row("Goals", [t["goals"] for t in teams]),
            row("Shots (on target)", [f"{t['shots']} ({t['shots_on_target']})" for t in teams]),
            row(
                "Passes completed / attempted",
                [
                    f"{t['passes']} / {t['passes_attempted']}"
                    + (f" ({100 * t['passes'] / t['passes_attempted']:.0f}%)" if t["passes_attempted"] else "")
                    for t in teams
                ],
            ),
            row("Average pass length", [_fmt(t["avg_pass_length_m"], " m", 1) for t in teams]),
            row("Ball won from opponent", [t["turnovers_won"] for t in teams]),
            row("Team width (avg)", [_fmt(t["width_m"], " m", 1) for t in teams]),
            row("Team depth (avg)", [_fmt(t["depth_m"], " m", 1) for t in teams]),
        ]
    )

    events = (
        "".join(
            f"<li><button class=seek data-t='{e['frame'] / fps:.2f}'>{_clock(e['frame'], fps)}</button> "
            f"<span class=sw style='background:{colours[e['team']] if e['team'] is not None else '#888'}'></span>"
            f"<b>{html.escape(e['kind'].capitalize())}</b> "
            f"{html.escape(e['player'] or ('Team ' + str(e['team'] + 1) if e['team'] is not None else ''))}"
            f"{_speed(e)}</li>"
            for e in stats["events"]
        )
        or "<li class=muted>No shots or goals detected.</li>"
    )

    quality = stats.get("calibration_quality", 1.0)
    movement = quality >= MIN_CALIBRATION_QUALITY
    headers = (
        ["ID", "On screen (s)"]
        + (["Distance (m)", "m/min", "Sprint speed (km/h)"] if movement else [])
        + ["Touches", "Passes", "Possession (s)", "Shots", "Goals"]
    )

    def cells(p):
        values = [html.escape(p["label"]), f"{p['seconds']:.0f}"]
        if movement:
            per_min = p["distance_m"] / p["seconds"] * 60 if p["seconds"] else 0
            values += [f"{p['distance_m']:.0f}", f"{per_min:.0f}", f"{p['top_speed_kmh']:.1f}"]
        values += [p["touches"], p["passes"], f"{p['possession_s']:.1f}", p["shots"], p["goals"]]
        return "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"

    player_tables = "".join(
        f"<h3><span class=sw style='background:{colours[k]}'></span>Team {k + 1}</h3><div class=scroll><table>"
        + "<tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr>"
        + "".join(cells(p) for p in stats["players"] if p["team"] == k)
        + "</table></div>"
        for k in (0, 1)
    )
    movement_note = (
        ""
        if movement
        else (
            f"<p class=muted>Distance and speed are hidden: only {100 * quality:.0f}% of calibrated frames have pitch "
            "keypoints that agree with each other, so positions can jump by several metres. They are still in "
            "stats.json.</p>"
        )
    )

    video_name = Path(stats["output_video"]).name
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Analysis Report</title>
<style>
:root {{ --bg:#f6f7f5; --card:#fff; --text:#1c1f1c; --muted:#5d665d; --line:#dde2dd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141714; --card:#1e221e; --text:#e8ece8; --muted:#9aa49a; --line:#323832; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,sans-serif; }}
main {{ max-width:1040px; margin:auto; padding:24px 16px; }}
h1 {{ margin:0 0 4px; }} h2 {{ margin-top:36px; }}
.muted {{ color:var(--muted); }}
.score {{ font-size:34px; font-weight:700; margin:16px 0; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.card b {{ display:block; font-size:19px; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
.sw {{ display:inline-block; width:12px; height:12px; border:1px solid var(--muted); border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
img, video {{ max-width:100%; border-radius:8px; }} img {{ background:#fff; }}
ul.events {{ list-style:none; padding:0; }} ul.events li {{ padding:4px 0; }}
button.seek {{ font:inherit; font-variant-numeric:tabular-nums; border:1px solid var(--line); background:var(--card); color:var(--text); border-radius:6px; padding:2px 8px; cursor:pointer; min-height:32px; }}
</style></head><body><main>
<h1>Match analysis</h1>
<div class=muted>{html.escape(stats["video"])}</div>
<div class=score><span class=sw style='background:{colours[0]};width:22px;height:22px'></span>Team 1
  <span>{teams[0]["goals"]} – {teams[1]["goals"]}</span>Team 2<span class=sw style='background:{colours[1]};width:22px;height:22px'></span></div>
<div class=cards>{"".join(f"<div class=card><span class=muted>{k}</span><b>{v}</b></div>" for k, v in cards)}</div>

<h2>Video</h2>
<video id=video src="{html.escape(video_name)}" controls preload=metadata></video>

<h2>Teams</h2>
<div class=scroll><table><tr><th></th><th><span class=sw style='background:{colours[0]}'></span>Team 1</th>
<th><span class=sw style='background:{colours[1]}'></span>Team 2</th></tr>{team_table}</table></div>

<h2>Shots and goals</h2>
<ul class=events>{events}</ul>
<div class=grid>{shot_map(stats, colours)}</div>

<h2>Possession over time</h2>
{possession_chart(stats, colours)}

<h2>Where the play happened</h2>
<div class=grid>{heatmap_chart(teams[0]["heatmap"], "Team 1 positions", colours[0])}
{heatmap_chart(teams[1]["heatmap"], "Team 2 positions", colours[1])}
{heatmap_chart(stats["ball_heatmap"], "Ball", "#ffffff")}</div>

<h2>Players</h2>
<p class=muted>IDs 1–10 are outfield slots per team. A slot passes to the nearest new track when the tracker loses a
player, so across camera cuts one ID can cover different people.</p>
{movement_note}
{player_tables}

<h2>How to read this</h2>
<ul class=muted>
<li>Player, goalkeeper, referee and ball detection uses football-trained YOLOv8 models; the pitch keypoint model maps
the image onto a 120 × 70 m pitch. Distances, speeds, heatmaps, shots and goals use only calibrated frames.</li>
<li>Only people and balls inside the touch and goal lines count. One referee and one goalkeeper per end are kept per frame.</li>
<li>Possession goes to the player within 1.5 m of the ball. A pass is the ball moving between two players of a team in one
camera shot.</li>
<li>A shot is the ball leaving a player at 43 km/h or more towards a goal from within 30 m. Shot speed is measured on
the ground plane, so a lofted shot reads faster than it was. A goal is the ball seen inside the
goal mouth; shots shown in close-up or from behind the goal are missed, and a high shot over the bar can be miscounted.</li>
<li>The same goal shown again within 60 s (replays) counts once, but replayed shots still count again.</li>
</ul>
</main>
<script>
document.querySelectorAll("button.seek").forEach(b => b.addEventListener("click", () => {{
  const v = document.getElementById("video"); v.currentTime = +b.dataset.t; v.play(); v.scrollIntoView({{behavior: "smooth"}});
}}));
</script>
</body></html>"""
    out = stats_path.with_name("report.html")
    out.write_text(page)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python report.py output/<video>/stats.json")
    print(write_report(sys.argv[1]))
