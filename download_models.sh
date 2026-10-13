#!/usr/bin/env bash
# Football-trained YOLOv8 weights from roboflow/sports (examples/soccer/setup.sh), ~140 MB each.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
get() { [[ -s "models/$1" ]] || uvx gdown -O "models/$1" "https://drive.google.com/uc?id=$2"; }
get football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
get football-ball-detection.pt 1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V
get football-pitch-detection.pt 1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
