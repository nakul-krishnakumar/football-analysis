# Football analysis

Turns a broadcast football video into an annotated video, a stats file and an HTML match report.

```sh
./download_models.sh                                  # once: ~400 MB of football-trained weights
uv run python main.py --video_path input/football.mp4  # --device cpu if the GPU is unstable
```

Output goes to `output/<video name>/`:
- `annotated.mp4`: team-coloured player IDs (1–10, `GK`, `REF`), the ball, a marker over the player with the ball,
  live score and possession, and a minimap. Re-encoded to H.264 when `ffmpeg` is installed, so browsers can play it.
- `stats.json`: every number in the report.
- `report.html`: score, team comparison, clickable shot/goal timeline, shot map, heatmaps, per-player table.
  Rebuild it without re-running anything: `uv run python report.py output/<video name>/stats.json`.

Model inference is cached in `output/cache/<video name>.pkl` and checkpointed every 250 frames: a killed run resumes,
and a re-run with the same video and models skips straight to the analysis (seconds instead of hours on CPU).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Detect | `detect.py` | Player/goalkeeper/referee/ball YOLOv8, a dedicated ball model, and a 32-keypoint pitch model, run on frames where grass fills ≥25% of the image. |
| Pitch | `pitch.py` | Keypoints → image-to-pitch homography (120 × 70 m), median-smoothed over ±5 frames inside each camera shot. |
| Track | `track.py` | Keeps only people inside the touch and goal lines; ByteTrack per camera shot; teams by jersey-colour clustering; roles from model votes overruled by kit colour; one referee and one goalkeeper per end per frame; assistant referees on the touchline dropped; stable IDs 1–10 per team. |
| Events | `events.py` | Ball candidates filtered (inside the lines or in a goal, not a flat painted spot, not on a penalty/centre spot unless a player is near, consistent with the ball's path); possession, passes, turnovers, shots, goals, distance and sprint speed. |
| Render / report | `render.py`, `report.py` | Annotated video and HTML report. |

## Accuracy notes

Checked by hand against `input/football.mp4` (Real Madrid v Villarreal highlights, 3:09):

| | Detected | In the footage |
|---|---|---|
| Goals | 1 (Villarreal, 1:18, long-range shot by yellow #6) | 2: the other (Madrid, ~2:01) is shown only in close-up |
| Shots | 3, all real Madrid/Villarreal attempts | more; close-ups and behind-goal angles are not measurable |
| Referee | the referee in every wide shot; linesmen ignored | |
| Goalkeepers | Villarreal's keeper at whichever end he defends | Madrid's keeper is rarely in view |

- **Pitch calibration is the limit.** The keypoint model was trained on Bundesliga footage; on this broadcast it often
  mislabels lines (e.g. reads the penalty-box line as the halfway line), so positions can jump by metres. Every run
  measures this as `calibration_quality` (share of calibrated frames whose keypoints agree with one homography).
  Below 50% the report hides distance and speed (still in `stats.json`). This clip scores 10%. Positions are still
  good enough for possession, the touch-line filter, shots, goals and heatmaps.
  Next step if needed: refine each homography against the white pitch lines in the image.
- **IDs 1–10:** when the tracker loses a player, the new track takes the free ID whose last pitch position is nearest.
  This holds within continuous play; after a camera cut it is a best guess, so per-ID stats over a highlights reel
  mix people. Jersey-number re-ID is not attempted: numbers are a few pixels tall from the main camera.
- **Shots and goals** need a calibrated wide shot. A shot is credited to the team attacking that goal (known from where
  each goalkeeper stands, so a half-time switch is followed); a defender's touch just before is treated as a block.
  Shot speed is measured on the ground plane, so lofted shots read high (the report shows "lofted" above 140 km/h).
  Replays of the same goal within 60 s count once; replayed shots count again.
- **Sprint speed** is the 95th percentile of a player's one-second speeds, not the single maximum, so one calibration
  glitch cannot set it.

## GPU

On this machine the NVIDIA driver resets the GPU under sustained load (`NVRM ... GPU_IN_FULLCHIP_RESET` in the kernel
log), so runs used `--device cpu`: about 2.5 s per pitch frame for detection, about 1 minute for everything after it.

## Tests

```sh
uv run pytest
```
