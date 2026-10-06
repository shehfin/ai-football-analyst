import matplotlib.pyplot as plt
import numpy as np


def generate_heatmap(df, player_id, frame_width, frame_height, save_path):
    """
    Generate and save a heatmap image showing where a specific player
    spent their time on screen, based on their tracked (x, y) positions.
    """
    player_df = df[df["player_id"] == player_id]

    if player_df.empty:
        raise ValueError(f"No position data found for player ID {player_id}")

    xs = player_df["x"].to_numpy()
    ys = player_df["y"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2D histogram: counts how many times the player was in each grid cell
    heatmap, xedges, yedges = np.histogram2d(
        xs, ys, bins=[30, 20],
        range=[[0, frame_width], [0, frame_height]],
    )

    # Transpose + flip so it displays with (0,0) at top-left like a video frame
    ax.imshow(
        heatmap.T, origin="upper", cmap="hot",
        extent=[0, frame_width, frame_height, 0],
        aspect="auto", alpha=0.85,
    )

    ax.set_title(f"Player {player_id} — Estimated Movement Heatmap")
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

    return save_path
