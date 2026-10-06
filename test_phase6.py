from src.ball_tracking import detect_ball, interpolate_ball_positions
from src.event_detection import detect_events
from src.video_processing import VideoError, get_video_info, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    info = get_video_info(VIDEO_PATH)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]

    ball_rows = []
    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        ball = detect_ball(frame)
        if ball is not None:
            x1, y1, x2, y2 = ball["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ball_rows.append({"frame": frame_index, "x": cx, "y": cy, "detected": True})
        else:
            ball_rows.append({"frame": frame_index, "x": None, "y": None, "detected": False})

        if frame_index % 60 == 0:
            print(f"Processed frame {frame_index} (t={timestamp:.2f}s)")

    filled = interpolate_ball_positions(ball_rows)

    events = detect_events(filled, fps, width, height)

    print(f"\nDetected {len(events)} events:")
    for event in events:
        print(f"  [{event['timestamp']:.2f}s] {event['type'].upper()} -- {event['confidence_note']}")

    if not events:
        print("  (No events detected -- this can happen with low ball detection "
              "coverage, or if the rule thresholds don't match this clip's camera "
              "distance/speed scale. See notes below.)")

except VideoError as error:
    print(f"Video problem: {error}")
    