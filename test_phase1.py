import cv2

from src.video_processing import VideoError, get_video_info, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    # 1. Print basic info
    info = get_video_info(VIDEO_PATH)
    print("Video info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 2. Read the first 150 frames, save every 30th as an image
    saved = 0
    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        if frame_index >= 150:
            break

        if frame_index % 30 == 0:
            filename = f"data/processed/frame_{frame_index:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename} (time = {timestamp:.2f}s, shape = {frame.shape})")
            saved += 1

    print(f"Done. Saved {saved} frames to data/processed/")

except VideoError as error:
    print(f"Video problem: {error}")