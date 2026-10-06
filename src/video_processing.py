from pathlib import Path

import cv2


class VideoError(Exception):
    """Raised when a video cannot be opened or read."""


def open_video(video_path):
    """Open a video file and return an OpenCV VideoCapture object."""
    path = Path(video_path)

    if not path.exists():
        raise VideoError(f"File not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise VideoError(
            f"Could not open '{path.name}'. The file may be corrupted "
            "or in an unsupported format."
        )
    return cap


def get_video_info(video_path):
    """Return basic information about a video as a dictionary."""
    cap = open_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # always release the video when done

    if fps <= 0 or total_frames <= 0:
        raise VideoError("Video metadata is invalid (FPS or frame count is 0).")

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(total_frames / fps, 2),
    }


def read_frames(video_path, every_n=1):
    """
    Generator that yields (frame_index, timestamp_seconds, frame).
    every_n=1 gives every frame; every_n=5 gives every 5th frame.
    """
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:  # no more frames (or a read error)
            break
        if frame_index % every_n == 0:
            yield frame_index, frame_index / fps, frame
        frame_index += 1

    cap.release()


def get_frame_at_time(video_path, seconds):
    """Return the frame at a given time (in seconds), or raise VideoError."""
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = min(int(seconds * fps), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # jump to that frame
    success, frame = cap.read()
    cap.release()

    if not success:
        raise VideoError(f"Could not read a frame at {seconds} seconds.")
    return frame 