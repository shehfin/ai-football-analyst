from ultralytics import YOLO

PERSON_CLASS_ID = 0

# Reuse the same small model, but now for tracking instead of plain detection
_model = YOLO("yolov8n.pt")


def track_players(frame, confidence_threshold=0.4):
    """
    Run YOLO + ByteTrack on a single frame and return tracked players.

    IMPORTANT: Call this on frames IN ORDER (frame 0, then 1, then 2...).
    The tracker keeps internal memory between calls to match players
    across frames. Skipping frames or calling out of order breaks tracking.

    Each tracked player is a dictionary:
        {
            "id": 7,                     # stable ID across frames (None if not yet assigned)
            "box": (x1, y1, x2, y2),
            "confidence": 0.82
        }
    """
    results = _model.track(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=confidence_threshold,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    tracked_players = []
    for result in results:
        if result.boxes.id is None:
            # No tracks confirmed yet (can happen on the very first frame)
            continue

        boxes = result.boxes.xyxy.tolist()
        ids = result.boxes.id.tolist()
        confidences = result.boxes.conf.tolist()

        for box, track_id, confidence in zip(boxes, ids, confidences):
            x1, y1, x2, y2 = box
            tracked_players.append({
                "id": int(track_id),
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": round(float(confidence), 2),
            })

    return tracked_players


def draw_tracks(frame, tracked_players):
    """Draw boxes with player IDs (instead of just confidence) on a copy of the frame."""
    import cv2

    annotated = frame.copy()
    for player in tracked_players:
        x1, y1, x2, y2 = player["box"]
        player_id = player["id"]

        color = (
            (player_id * 47) % 256,
            (player_id * 91) % 256,
            (player_id * 13) % 256,
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {player_id}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    return annotated