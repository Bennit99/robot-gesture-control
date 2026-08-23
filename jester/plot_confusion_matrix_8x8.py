import os

import matplotlib.pyplot as plt
import numpy as np


def generate_and_plot_8x8():
    cm_path = "jester_runs/gru-1/eval/confusion_matrix.npy"
    cm_full = np.load(cm_path)

    class_names_full = [
        "Swiping Left",
        "Swiping Right",
        "Swiping Down",
        "Swiping Up",
        "Pushing Hand Away",
        "Pulling Hand In",
        "Sliding Two Fingers Left",
        "Sliding Two Fingers Right",
        "Sliding Two Fingers Down",
        "Sliding Two Fingers Up",
        "Pushing Two Fingers Away",
        "Pulling Two Fingers In",
        "Rolling Hand Forward",
        "Rolling Hand Backward",
        "Turning Hand Clockwise",
        "Turning Hand Counterclockwise",
        "Zooming In With Full Hand",
        "Zooming Out With Full Hand",
        "Zooming In With Two Fingers",
        "Zooming Out With Two Fingers",
        "Thumb Up",
        "Thumb Down",
        "Shaking Hand",
        "Stop Sign",
        "Drumming Fingers",
        "No gesture",
        "Doing other things",
    ]
    # NOTE: The classification report showed:
    # 23: Thumb Up
    # 24: Thumb Down
    # 25: Shaking Hand
    # 26: Stop Sign
    # 27: Drumming Fingers
    # 28: No gesture
    # 29: Doing other things
    # Wait, 1-indexed in report vs 0-indexed in python.
    # 0: Swiping Left, ..., 26: Doing other things. Total 27 classes.

    # Exact indices for the subset:
    # Swiping Left (0)
    # Swiping Right (1)
    # Turning Hand Clockwise (14)  Wait, 15 in my list?
    # Let's dynamically find the indices to be safe!

    subset_names = [
        "Turning Hand Clockwise",
        "Turning Hand Counterclockwise",
        "Swiping Left",
        "Swiping Right",
        "Drumming Fingers",
        "Thumb Down",
        "No gesture",
        "Doing other things",
    ]

    indices = [class_names_full.index(name) for name in subset_names]

    # Normalize full matrix first (row normalization / recall)
    row_sums = cm_full.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_full, row_sums, where=row_sums != 0, out=np.zeros_like(cm_full, dtype=float)
    )

    # Extract 8x8 subset
    cm_8x8 = cm_norm[np.ix_(indices, indices)]

    old_size = plt.rcParams.get("font.size")
    plt.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    im = ax.imshow(cm_8x8, interpolation="nearest", cmap="Blues", aspect="auto")

    # We want a colorbar that scales with the axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    # Tick labels
    ax.set_xticks(np.arange(len(subset_names)))
    ax.set_yticks(np.arange(len(subset_names)))
    ax.set_xticklabels(subset_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(subset_names, fontsize=8)

    # Text annotations
    thresh = cm_8x8.max() / 2.0
    for i in range(len(subset_names)):
        for j in range(len(subset_names)):
            val = cm_8x8[i, j]
            if val == 0:
                continue
            text = f"{val:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=7,
            )

    ax.set_xlabel("Predicted Gesture")
    ax.set_ylabel("Actual Gesture")

    out_dir = "MA/latex/images"
    os.makedirs(out_dir, exist_ok=True)

    out_png = os.path.join(out_dir, "gru_confusion_matrix_8x8.png")
    out_pgf = os.path.join(out_dir, "gru_confusion_matrix_8x8.pgf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_pgf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    plt.rcParams.update({"font.size": old_size})
    print(f"Saved {out_png} and .pgf")


if __name__ == "__main__":
    generate_and_plot_8x8()
