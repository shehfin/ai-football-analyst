from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.ball_tracking import detect_ball, interpolate_ball_positions
from src.event_detection import detect_events
from src.heatmap import generate_heatmap
from src.player_analysis import box_to_point, calculate_player_stats, positions_to_dataframe
from src.tracking import track_players
from src.video_processing import VideoError, get_video_info, read_frames

UPLOAD_DIR = Path("data/raw_videos")
HEATMAP_DIR = Path("output/heatmaps")
REPORTS_DIR = Path("output/reports")
for d in (UPLOAD_DIR, HEATMAP_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_DURATION_SECONDS = 5 * 60

st.set_page_config(page_title="AI Football Video Analyst", layout="wide")
st.title("AI Football Video Analyst")
st.caption(
    "A computer vision pipeline for football videos: player detection, tracking, "
    "estimated movement analytics, heatmaps, ball tracking, and rule-based event detection."
)

# ---------- Upload ----------

uploaded_file = st.file_uploader(
    "Upload a football video (5 minutes or shorter)",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is None:
    st.info("Upload a video to get started.")
    st.stop()

video_path = UPLOAD_DIR / Path(uploaded_file.name).name
with open(video_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

try:
    info = get_video_info(video_path)
except VideoError as error:
    st.error(f"Could not read this video: {error}")
    st.stop()

if info["duration_seconds"] > MAX_DURATION_SECONDS:
    st.error(
        f"This video is {info['duration_seconds']:.1f} seconds long. "
        f"Please upload a video of {MAX_DURATION_SECONDS // 60} minutes or shorter."
    )
    st.stop()

st.subheader("Video information")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Resolution", f"{info['width']}x{info['height']}")
col2.metric("FPS", info["fps"])
col3.metric("Total frames", info["total_frames"])
col4.metric("Duration (s)", info["duration_seconds"])

st.video(str(video_path))

# ---------- Run pipeline ----------

run_clicked = st.button("Analyze video", type="primary")

if run_clicked:
    fps = info["fps"]
    width = info["width"]
    height = info["height"]

    position_rows = []
    ball_rows = []

    progress = st.progress(0.0, text="Processing frames...")
    total_frames = info["total_frames"]

    for frame_index, timestamp, frame in read_frames(video_path):
        tracked = track_players(frame)
        for player in tracked:
            x, y = box_to_point(player["box"])
            position_rows.append({
                "frame": frame_index, "timestamp": timestamp,
                "player_id": player["id"], "x": x, "y": y,
            })

        ball = detect_ball(frame)
        if ball is not None:
            x1, y1, x2, y2 = ball["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ball_rows.append({"frame": frame_index, "x": cx, "y": cy, "detected": True})
        else:
            ball_rows.append({"frame": frame_index, "x": None, "y": None, "detected": False})

        if frame_index % 10 == 0:
            progress.progress(min(frame_index / total_frames, 1.0),
                               text=f"Processing frame {frame_index}/{total_frames}")

    progress.progress(1.0, text="Calculating stats...")

    df = positions_to_dataframe(position_rows)
    stats_df = calculate_player_stats(df, fps)
    stats_df.to_csv(REPORTS_DIR / "player_stats.csv", index=False)

    filled_ball = interpolate_ball_positions(ball_rows)
    events = detect_events(filled_ball, fps, width, height)

    progress.empty()

    # Save results into session_state so tab switches don't re-run the pipeline
    st.session_state["df"] = df
    st.session_state["stats_df"] = stats_df
    st.session_state["events"] = events
    st.session_state["ball_detection_rate"] = round(
        100 * sum(r["detected"] for r in ball_rows) / len(ball_rows), 1
    )
    st.session_state["width"] = width
    st.session_state["height"] = height
    st.success("Analysis complete.")

# ---------- Display results ----------

if "stats_df" in st.session_state:
    df = st.session_state["df"]
    stats_df = st.session_state["stats_df"]
    events = st.session_state["events"]

    tab_players, tab_heatmap, tab_events, tab_summary = st.tabs(
        ["Player statistics", "Heatmap", "Events", "Summary"]
    )

    with tab_players:
        st.caption(
            "Distance and speed are ESTIMATED from pixel movement using a rough "
            "meters-per-pixel assumption -- not precise GPS-level measurements. "
            "Implausible frame-to-frame jumps (likely tracking ID errors) are "
            "excluded; see `flagged_jumps`."
        )
        st.dataframe(stats_df, use_container_width=True)

    with tab_heatmap:
        player_ids = stats_df["player_id"].tolist()
        if player_ids:
            selected_id = st.selectbox("Select a player ID", player_ids)
            if st.button("Generate heatmap"):
                save_path = HEATMAP_DIR / f"player_{int(selected_id)}_heatmap.jpg"
                generate_heatmap(
                    df, selected_id,
                    st.session_state["width"], st.session_state["height"],
                    str(save_path),
                )
                st.image(str(save_path), use_container_width=True)
        else:
            st.info("No players with enough tracked frames to generate a heatmap.")

    with tab_events:
        st.caption(
            "Event detection uses simple rule-based heuristics on ball speed and "
            "position, not a trained model. Accuracy is limited, especially for "
            "shots and goals -- see README for details."
        )
        st.metric("Ball detection rate (raw)", f"{st.session_state['ball_detection_rate']}%")
        if events:
            events_df = pd.DataFrame(events)
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No events detected for this video.")

    with tab_summary:
        st.subheader("Summary statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Players tracked", len(stats_df))
        col2.metric("Events detected", len(events))
        col3.metric("Ball detection rate", f"{st.session_state['ball_detection_rate']}%")

        if not stats_df.empty:
            top_runner = stats_df.iloc[0]
            st.write(
                f"Most ground covered (estimated): **Player {int(top_runner['player_id'])}** "
                f"-- {top_runner['distance_estimated_m']}m over "
                f"{top_runner['time_visible_s']}s visible."
            )