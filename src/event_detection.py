import math

# Speed thresholds in pixels/frame -- rough, tune based on testing.
# These depend on video resolution and camera distance, so they're
# approximate, not physically calibrated.
PASS_SPEED_THRESHOLD = 15
SHOT_SPEED_THRESHOLD = 28

# How close (in pixels) the ball must be to a goal region to count
# as "near goal" for shot/goal classification.
GOAL_PROXIMITY_PX = 120


def define_goal_regions(frame_width, frame_height):
    """
    Rough goal-mouth regions near the left and right edges of the frame,
    assuming a standard wide broadcast angle where both goals are near
    the horizontal edges. This is an approximation -- for a precise
    setup, these regions should be calibrated per-video.
    """
    goal_height = frame_height * 0.3
    goal_top = (frame_height - goal_height) / 2

    left_goal = {
        "x_min": 0, "x_max": GOAL_PROXIMITY_PX,
        "y_min": goal_top, "y_max": goal_top + goal_height,
    }
    right_goal = {
        "x_min": frame_width - GOAL_PROXIMITY_PX, "x_max": frame_width,
        "y_min": goal_top, "y_max": goal_top + goal_height,
    }
    return left_goal, right_goal


def _is_near_goal(x, y, goal_region):
    return (goal_region["x_min"] <= x <= goal_region["x_max"] and
            goal_region["y_min"] <= y <= goal_region["y_max"])


def detect_events(ball_positions, fps, frame_width, frame_height):
    """
    Given a list of ball positions over time (from Phase 5, after
    interpolation), return a list of detected events using simple
    rule-based heuristics.

    ball_positions: list of dicts, each with at least
        {"frame": i, "x": cx, "y": cy} (x/y can be None if unknown)

    Returns a list of event dicts:
        {"frame": i, "timestamp": t, "type": "pass"/"shot"/"goal", "confidence_note": "..."}

    IMPORTANT: this is a simple heuristic system based on ball speed and
    position, NOT a trained model. It will miss real events and sometimes
    flag non-events. It's a placeholder designed to be replaced later by
    an ML classifier trained on labeled event data -- see README.
    """
    left_goal, right_goal = define_goal_regions(frame_width, frame_height)
    events = []

    known_points = [p for p in ball_positions if p["x"] is not None]

    for i in range(1, len(known_points)):
        prev = known_points[i - 1]
        curr = known_points[i]

        frame_gap = curr["frame"] - prev["frame"]
        if frame_gap <= 0:
            continue

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        distance = math.hypot(dx, dy)
        speed = distance / frame_gap  # pixels per frame, gap-normalized

        near_left_goal = _is_near_goal(curr["x"], curr["y"], left_goal)
        near_right_goal = _is_near_goal(curr["x"], curr["y"], right_goal)
        near_goal = near_left_goal or near_right_goal

        timestamp = curr["frame"] / fps

        if speed >= SHOT_SPEED_THRESHOLD and near_goal:
            events.append({
                "frame": curr["frame"],
                "timestamp": round(timestamp, 2),
                "type": "shot",
                "confidence_note": "high ball speed near goal area",
            })

            # A shot that ends with the ball settling inside the goal
            # region for a few consecutive frames is flagged as a goal.
            lookahead = known_points[i:i + 5]
            if len(lookahead) >= 3 and all(
                _is_near_goal(p["x"], p["y"], left_goal if near_left_goal else right_goal)
                for p in lookahead
            ):
                events.append({
                    "frame": curr["frame"],
                    "timestamp": round(timestamp, 2),
                    "type": "goal",
                    "confidence_note": "ball remained in goal area after shot-like movement",
                })

        elif speed >= PASS_SPEED_THRESHOLD and not near_goal:
            events.append({
                "frame": curr["frame"],
                "timestamp": round(timestamp, 2),
                "type": "pass",
                "confidence_note": "moderate ball speed change, open play",
            })

    return deduplicate_events(events)


def deduplicate_events(events, min_gap_seconds=1.5):
    """
    Collapse events of the same type that are very close together in time
    into a single event -- otherwise a single real pass/shot can trigger
    multiple nearby frame detections due to frame-by-frame speed noise.
    """
    if not events:
        return events

    events = sorted(events, key=lambda e: e["timestamp"])
    deduped = [events[0]]

    for event in events[1:]:
        last = deduped[-1]
        same_type = event["type"] == last["type"]
        close_in_time = (event["timestamp"] - last["timestamp"]) < min_gap_seconds

        if same_type and close_in_time:
            continue  # skip, treat as the same event already recorded
        deduped.append(event)

    return deduped