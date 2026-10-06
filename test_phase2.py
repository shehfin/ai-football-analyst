import cv2

from src.detection import detect_players, draw_detections
from src.video_processing import VideoError, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    frame_count = 0
    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        if frame_index >= 90:  # test on the first 3 seconds at 30fps
            break

        if frame_index % 30 == 0:  # test one frame per second
            detections = detect_players(frame)
            print(f"Frame {frame_index} (t={timestamp:.2f}s): {len(detections)} players detected")
            for i, det in enumerate(detections):
                print(f"  Player {i}: box={det['box']}, confidence={det['confidence']}")

            annotated = draw_detections(frame, detections)
            filename = f"data/processed/detected_{frame_index:04d}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"  Saved {filename}")
            frame_count += 1

    print(f"Done. Processed {frame_count} frames.")

except VideoError as error:
    print(f"Video problem: {error}")