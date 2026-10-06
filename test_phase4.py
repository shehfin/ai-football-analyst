from src.heatmap import generate_heatmap
from src.player_analysis import box_to_point, calculate_player_stats, positions_to_dataframe
from src.tracking import track_players
from src.video_processing import VideoError, get_video_info, read_frames

VIDEO_PATH = "data/raw_videos/sample.mp4"

try:
    info = get_video_info(VIDEO_PATH)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]

    position_rows = []

    for frame_index, timestamp, frame in read_frames(VIDEO_PATH):
        tracked = track_players(frame)

        for player in tracked:
            x, y = box_to_point(player["box"])
            position_rows.append({
                "frame": frame_index,
                "timestamp": timestamp,
                "player_id": player["id"],
                "x": x,
                "y": y,
            })

        if frame_index % 30 == 0:
            print(f"Processed frame {frame_index} (t={timestamp:.2f}s)")

    df = positions_to_dataframe(position_rows)
    df.to_csv("data/processed/positions.csv", index=False)
    print(f"\nSaved {len(df)} position rows to data/processed/positions.csv")

    stats_df = calculate_player_stats(df, fps)
    stats_df.to_csv("output/reports/player_stats.csv", index=False)
    print("\nPlayer stats (top 10 by estimated distance):")
    print(stats_df.head(10).to_string(index=False))

    # Generate a heatmap for whichever player has the most tracked frames
    top_player_id = stats_df.sort_values("frames_tracked", ascending=False).iloc[0]["player_id"]
    heatmap_path = f"output/heatmaps/player_{int(top_player_id)}_heatmap.jpg"
    generate_heatmap(df, top_player_id, width, height, heatmap_path)
    print(f"\nSaved heatmap for player {int(top_player_id)} to {heatmap_path}")

except VideoError as error:
    print(f"Video problem: {error}")
    