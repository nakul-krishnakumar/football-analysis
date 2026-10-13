# AGENTS.md

Broadcast football video → annotated video, `stats.json`, HTML report. See README.md for what it does and its known limits.

## Commands

```sh
./download_models.sh                                              # weights into models/ (gitignored)
uv run python main.py --video_path input/football.mp4 --device cpu
uv run python report.py output/football/stats.json                # rebuild report only
uv run pytest                                                     # tests/test_pipeline.py
uvx ruff format *.py tests/ && uvx ruff check --select E,F,W,B --ignore B905,E501 *.py tests/
```

## Layout

One file per stage, called in order by `main.py`:
`detect.py` (models → cache) → `pitch.py` (homography) → `track.py` (people, teams, roles, IDs 1–10)
→ `events.py` (ball, possession, passes, shots, goals, movement) → `render.py` / `report.py`.
Tuning constants sit at the top of each module with a one-line reason; change them there.

## Rules that save hours

- **Never re-run detection casually.** Model inference is cached in `output/cache/<video>.pkl` (keyed on video
  path, mtime and model file sizes) and takes ~2 h per 3-min clip on CPU. Everything after it runs in ~1 min.
  Iterate on `track.py` / `events.py` against the cache.
- **Use the CPU.** This machine's GPU driver resets under sustained load. Run with `--device cpu` (or
  `CUDA_VISIBLE_DEVICES=""`) and put long jobs in the background.
- **Check changes against the footage, not just the numbers.** Pull frames from `output/<video>/annotated.mp4`
  around an event (cv2, a few tiles in a grid) and look. Most bugs fixed so far (referee labelled as a player,
  shots credited to the wrong team, a spare ball hijacking the ball track) were only visible that way.
- **Pitch calibration is the weak link.** The keypoint model mislabels lines on non-Bundesliga broadcasts, so
  positions can jump by metres. Don't trust per-frame speeds; keep anything speed-like robust (percentiles,
  displacement over ≥1 s) and keep the `calibration_quality` gate in the report.
- New detection rules get a test in `tests/test_pipeline.py`; tests build synthetic frames and homographies,
  so they need no models or video.

## Conventions

- Pitch coordinates are metres on a 120 × 70 template (`pitch.py`): x along the length, goals at x=0 and x=120.
- Teams are `0`/`1` (from jersey-colour clustering, order arbitrary); labels are `"1"`–`"10"`, `"GK"`, `"REF"`, `"?"`.
- Known shortcuts are marked `# shortcut: <limit>, <when to upgrade>`.
- `models/`, `output/` and `*.pt` are gitignored; don't commit weights or run outputs.
