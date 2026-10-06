import math

import pandas as pd

# Rough pixel-to-meter conversion for a typical wide broadcast shot.
# This is a ballpark constant, not a calibrated measurement -- see README.
PIXELS_PER_METER = 10.0

# A real player realistically moves at most ~38 km/h sprinting.
# 38 km/h = ~10.6 m/s = ~0.42m per frame at 25fps = ~4.2 pixels/frame
# at our PIXELS_PER_METER scale. We allow some margin for camera motion/
# measurement noise, so we cap at ~12 pixels/frame rather than being exact.
MAX_PLAUSIBLE_PIXELS_PER_FRAME = 12

# Below this many tracked frames, stats are too noisy to be meaningful.
MIN_FRAMES_FOR_STATS = 15


def positions_to_dataframe(position_rows):
    return pd.DataFrame(position_rows)


def box_to_point(box):
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    y = y2
    return x, y


def calculate_player_stats(df, fps):
    """
    Given a DataFrame of positions (frame, timestamp, player_id, x, y),
    return a DataFrame of per-player stats. Implausible frame-to-frame
    jumps (likely tracking ID errors) are excluded from distance/speed.
    """
    stats = []

    for player_id, group in df.groupby("player_id"):
        group = group.sort_values("frame")
        frames_tracked = len(group)
        time_visible_s = frames_tracked / fps

        xs = group["x"].to_numpy()
        ys = group["y"].to_numpy()
        frame_nums = group["frame"].to_numpy()

        total_pixels = 0.0
        flagged_jumps = 0

        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            frame_gap = frame_nums[i] - frame_nums[i - 1]
            pixel_dist = math.hypot(dx, dy)

            # Scale the threshold if frames were skipped (gap > 1)
            max_allowed = MAX_PLAUSIBLE_PIXELS_PER_FRAME * max(frame_gap, 1)

            if pixel_dist <= max_allowed:
                total_pixels += pixel_dist
            else:
                flagged_jumps += 1  # likely an ID swap/occlusion -- skip it

        distance_m = total_pixels / PIXELS_PER_METER

        avg_speed_kmh = 0.0
        if time_visible_s > 0:
            avg_speed_kmh = (distance_m / time_visible_s) * 3.6

        stats.append({
            "player_id": player_id,
            "frames_tracked": frames_tracked,
            "time_visible_s": round(time_visible_s, 2),
            "distance_estimated_m": round(distance_m, 1),
            "avg_speed_estimated_kmh": round(avg_speed_kmh, 1),
            "flagged_jumps": flagged_jumps,
        })

    stats_df = pd.DataFrame(stats)

    # Only show reasonably well-tracked players in the leaderboard --
    # short-lived IDs are too noisy to report as meaningful stats
    reliable_df = stats_df[stats_df["frames_tracked"] >= MIN_FRAMES_FOR_STATS]

    return reliable_df.sort_values("distance_estimated_m", ascending=False).reset_index(drop=True)