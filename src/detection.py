from ultralytics import YOLO

# COCO class ID for "person" — YOLO's pretrained model already knows this class
PERSON_CLASS_ID = 0

# Load once, outside any function, so the model isn't reloaded every frame.
# "yolov8n.pt" is the smallest/fastest YOLO model — good for a beginner's CPU.
# The first time you run this, it downloads the weights automatically (~6MB).
_model = YOLO("yolov8n.pt")


def detect_players(frame, confidence_threshold=0.4):
    """
    Run YOLO on a single frame and return a list of detected players.

    Each detection is a dictionary:
        {
            "box": (x1, y1, x2, y2),   # pixel coordinates
            "confidence": 0.87          # how sure the model is
        }
    """
    # classes=[PERSON_CLASS_ID] tells YOLO to only look for people,
    # which is faster and avoids irrelevant objects (cars, etc.)
    results = _model(frame, classes=[PERSON_CLASS_ID], conf=confidence_threshold, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # corner coordinates
            confidence = float(box.conf[0])
            detections.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": round(confidence, 2),
            })

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and confidence scores on a copy of the frame."""
    import cv2

    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        confidence = det["confidence"]

        # Green rectangle around the player
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Confidence label above the box
        label = f"{confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    return annotated