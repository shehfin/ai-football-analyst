# AI Football Video Analyst

A computer vision pipeline that analyzes short football videos: detecting and tracking players, estimating movement and distance, generating heatmaps, tracking the ball, and flagging basic match events (passes, shots, goals) using rule-based heuristics. Built as a learning project to explore object detection, multi-object tracking, and sports analytics with Python.

Try it via a simple Streamlit dashboard — upload a video (5 minutes or shorter) and get back player stats, heatmaps, and a detected-events timeline.

## Demo

*(Add a screenshot or short GIF of the dashboard here before publishing.)*

## Features

- **Video upload & validation** — accepts MP4/MOV/AVI/MKV, rejects videos over 5 minutes
- **Player detection** — pretrained YOLOv8 (nano) detects players frame-by-frame with bounding boxes and confidence scores
- **Player tracking** — ByteTrack assigns a persistent ID to each player across frames
- **Movement analytics (estimated)** — per-player distance covered, average speed, time visible, and frames tracked
- **Heatmaps** — per-player movement distribution over the pitch
- **Ball detection & tracking** — generic YOLO ball detection, filtered for false positives, with short-gap interpolation
- **Event detection (rule-based)** — heuristic pass/shot/goal flagging based on ball speed and position
- **Streamlit dashboard** — tabbed UI for stats, heatmaps, events, and a summary

## Tech stack

Python · OpenCV · Ultralytics YOLOv8 · ByteTrack · NumPy · Pandas · Matplotlib · Streamlit

## Project structure

```
Football_AI/
├── data/
│   ├── raw_videos/       # uploaded videos (not tracked in git)
│   ├── processed/        # intermediate frames, position CSVs
│   └── event_dataset/    # reserved for future labeled event data
├── models/
│   ├── player_detector/  # for a custom player model, if trained later
│   └── ball_detector/    # for a football-specific ball model, if added later
├── src/
│   ├── video_processing.py   # video I/O, frame reading, metadata
│   ├── detection.py          # YOLO player detection
│   ├── tracking.py           # YOLO + ByteTrack player tracking
│   ├── player_analysis.py    # distance/speed/time-visible stats
│   ├── heatmap.py            # per-player heatmap generation
│   ├── ball_tracking.py      # ball detection + gap interpolation
│   └── event_detection.py    # rule-based pass/shot/goal detection
├── output/
│   ├── processed_video/
│   ├── heatmaps/
│   └── reports/
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourname/ai-football-video-analyst.git
cd ai-football-video-analyst

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Running it

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a short football video, and click **Analyze video**.

## Results on test footage

Tested on a ~28-second, 1280x720, 25fps match clip (build-up play ending in a goal).

| Metric | Result |
|---|---|
| Players detected per frame | 11–14 (wide match shot) |
| Detection confidence | Mostly 0.4–0.8 |
| Unique tracking IDs over 90 frames | 37 (vs. ~14 players visible) — see Known Limitations |
| Top estimated distance covered | ~145m over ~11.6s tracked |
| Raw ball detection rate | 14.1% of frames |
| Ball position coverage after short-gap interpolation | 45.2% of frames |
| Events detected | 7 passes; 0 shots/goals confidently classified |

## Known limitations

This is an MVP built with generic, off-the-shelf pretrained models — not models trained specifically on football footage. Several limitations are inherent to that choice, and are surfaced directly in the dashboard rather than hidden:

- **Tracking ID churn.** ByteTrack occasionally reassigns a new ID to a player after they're briefly occluded by another player, especially during fast, crowded passages of play. This means "player stats" sometimes represent a partial view of a real player's movement rather than their full trajectory. Implausible frame-to-frame position jumps (a signature of ID mix-ups) are detected and excluded from distance/speed calculations, with a `flagged_jumps` count shown per player for transparency.
- **Distance and speed are estimates, not measurements.** They're derived from pixel movement using a rough, uncalibrated meters-per-pixel assumption, since a broadcast camera's panning, zooming, and angle make true real-world distance impossible to recover without dedicated camera calibration. They're useful for comparing players relatively, not as precise GPS-level figures.
- **Ball detection is unreliable, especially at distance or with non-standard ball colors.** The generic YOLO model was not trained specifically to distinguish footballs from similarly shaped/colored pitch features (e.g., the penalty arc produced false positives before filtering was added). Detection rate drops further for balls that aren't the classic white/black pattern. A model trained on a football-specific dataset (e.g., SoccerNet) would likely perform substantially better.
- **Interpolated ball positions are estimates, not detections**, and are only generated for short gaps (≤1 second); longer gaps are left as unknown rather than guessed, since the ball's path isn't linear over longer spans.
- **Event detection is a simple heuristic, not a trained model.** It flags events based on ball speed and a fixed, uncalibrated guess at where the goal sits in frame. It reliably picks up pass-like ball movements but did not consistently classify shots or goals on test footage. It's intentionally isolated in `event_detection.py` so it can be replaced by a trained classifier later without touching the rest of the pipeline.
- **No YouTube URL support.** Scoped out of this MVP to keep the project focused on the core computer vision pipeline rather than download/format handling.

## Possible future improvements

- Train or fine-tune a player/ball detector on football-specific footage
- Add camera/pitch calibration for real-world distance and speed accuracy
- Improve tracking robustness with re-identification (jersey number or appearance-based matching) to reduce ID churn
- Replace rule-based event detection with a trained classifier on labeled pass/shot/goal clips
- Add YouTube URL support with duration validation

## Acknowledgements

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and its built-in ByteTrack integration.


