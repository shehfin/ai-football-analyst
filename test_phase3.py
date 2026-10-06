import cv2

from src.tracking import track_players, draw_tracks
from src.video_processing import VideoError, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    seen_ids = set()

    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        if frame_index >= 90:
            break

        # Call track_players on EVERY frame, in order, even if we only
        # save/print some of them -- this keeps the tracker's memory consistent.
        tracked = track_players(frame)

        current_ids = {p["id"] for p in tracked}
        seen_ids.update(current_ids)

        if frame_index % 30 == 0:
            print(f"Frame {frame_index} (t={timestamp:.2f}s): IDs = {sorted(current_ids)}")

            annotated = draw_tracks(frame, tracked)
            filename = f"data/processed/tracked_{frame_index:04d}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"  Saved {filename}")

    print(f"\nDone. Total unique player IDs seen across all frames: {sorted(seen_ids)}")

except VideoError as error:
    print(f"Video problem: {error}")
    