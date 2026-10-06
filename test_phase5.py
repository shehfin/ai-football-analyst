import cv2

from src.ball_tracking import detect_ball, draw_ball, interpolate_ball_positions
from src.video_processing import VideoError, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    ball_rows = []
    detected_count = 0

    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        ball = detect_ball(frame,confidence_threshold=0.05)

        if ball is not None:
            x1, y1, x2, y2 = ball["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ball_rows.append({"frame": frame_index, "x": cx, "y": cy, "detected": True})
            detected_count += 1
        else:
            ball_rows.append({"frame": frame_index, "x": None, "y": None, "detected": False})

        # Save a few sample frames with the ball marker drawn
        if frame_index % 30 == 0:
            annotated = draw_ball(frame, ball)
            filename = f"data/processed/ball_{frame_index:04d}.jpg"
            cv2.imwrite(filename, annotated)
            status = "BALL FOUND" if ball else "not found"
            print(f"Frame {frame_index} (t={timestamp:.2f}s): {status} -- saved {filename}")

    total_frames = len(ball_rows)
    detection_rate = (detected_count / total_frames) * 100
    print(f"\nRaw ball detection rate: {detected_count}/{total_frames} frames ({detection_rate:.1f}%)")

    filled = interpolate_ball_positions(ball_rows)
    filled_count = sum(1 for row in filled if row.get("interpolated"))
    final_known = sum(1 for row in filled if row["x"] is not None)
    final_rate = (final_known / total_frames) * 100
    print(f"After interpolation: {final_known}/{total_frames} frames have a ball position ({final_rate:.1f}%)")
    print(f"({filled_count} frames were filled in via interpolation)")

except VideoError as error:
    print(f"Video problem: {error}")