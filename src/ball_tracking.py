from ultralytics import YOLO

SPORTS_BALL_CLASS_ID = 32

# Separate model instance so ball detection logic is fully independent
# from player detection/tracking -- this makes it easy to swap in a
# football-specific model later (see README) without touching player code.
_ball_model = YOLO("yolov8n.pt")


def detect_ball(frame, confidence_threshold=0.35, max_box_size=22):
    """
    Detect the ball in a single frame using the generic pretrained model.

    max_box_size: reject detections wider or taller than this many pixels.
    A real ball stays small and roughly square in a broadcast shot; pitch
    markings (penalty arc, center circle) sometimes get misclassified as
    "sports ball" but tend to produce larger or more irregular boxes.
    """
    results = _ball_model(
        frame, classes=[SPORTS_BALL_CLASS_ID],
        conf=confidence_threshold, verbose=False,
    )

    best_detection = None
    best_confidence = 0.0

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            # Reject implausibly large or very non-square boxes --
            # a real ball is small and roughly as wide as it is tall
            if width > max_box_size or height > max_box_size:
                continue
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            if confidence > best_confidence:
                best_detection = {
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(confidence, 2),
                }
                best_confidence = confidence

    return best_detection


def interpolate_ball_positions(ball_rows, max_gap_frames=25):
    """
    Fill in SHORT gaps where the ball wasn't detected, via straight-line
    interpolation between the nearest known detections before and after
    the gap. Gaps longer than max_gap_frames are left unknown, since a
    long straight-line guess (e.g. across several seconds) is unreliable --
    the ball likely changed direction multiple times during that span.

    Default max_gap_frames=25 is roughly 1 second at 25fps.
    """
    filled = [dict(row) for row in ball_rows]
    n = len(filled)

    i = 0
    while i < n:
        if filled[i]["detected"]:
            i += 1
            continue

        gap_start = i
        while i < n and not filled[i]["detected"]:
            i += 1
        gap_end = i

        has_before = gap_start > 0
        has_after = gap_end < n
        gap_length = gap_end - gap_start

        if has_before and has_after and gap_length <= max_gap_frames:
            before = filled[gap_start - 1]
            after = filled[gap_end]

            for step, idx in enumerate(range(gap_start, gap_end), start=1):
                ratio = step / (gap_length + 1)
                filled[idx]["x"] = before["x"] + (after["x"] - before["x"]) * ratio
                filled[idx]["y"] = before["y"] + (after["y"] - before["y"]) * ratio
                filled[idx]["interpolated"] = True
        # else: gap too long (or at the very start/end) -- leave as unknown

    return filled


def draw_ball(frame, ball_detection):
    """Draw a marker on the ball's position, if detected."""
    import cv2

    if ball_detection is None:
        return frame

    annotated = frame.copy()
    x1, y1, x2, y2 = ball_detection["box"]
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    cv2.circle(annotated, center, 8, (0, 165, 255), 2)  # orange circle
    cv2.putText(
        annotated, f"ball {ball_detection['confidence']:.2f}",
        (center[0] + 10, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2,
    )
    return annotated