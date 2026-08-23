import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_both():
    t0 = time.time()
    cache_path = "jester/plot_cache/heatmap.json"

    if os.path.exists(cache_path):
        print("Loading cached aggregated heatmaps...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        h1_counts = np.array(data["h1_counts"])
        h2_counts = np.array(data["h2_counts"])
        xedges = np.array(data["xedges"])
        yedges = np.array(data["yedges"])
    else:
        print("Parsing parquets to compute centroids...")
        base_dir = "./data/jester/landmarks"
        centroid_list = []
        x_cols = [f"x{i}" for i in range(21)]
        y_cols = [f"y{i}" for i in range(21)]

        count = 0
        for root, _, files in os.walk(base_dir):
            if "landmarks.parquet" in files:
                path = os.path.join(root, "landmarks.parquet")
                try:
                    df = pd.read_parquet(path)
                    cx = df[x_cols].mean(axis=1).values
                    cy = df[y_cols].mean(axis=1).values
                    centroid_list.append(np.column_stack((cx, cy)))
                    count += 1
                    if count % 500 == 0:
                        print(f"Loaded {count} videos...")
                except Exception:
                    pass

        centroids = np.vstack(centroid_list)
        num_frames = len(centroids)

        print("Simulating augmentation...")
        np.random.seed(42)
        dist_factor = np.random.uniform(1.0, 4.0, size=(num_frames, 1))
        scale = 1.0 / dist_factor

        raw_shift_x = np.random.uniform(-0.5, 0.5, size=(num_frames, 1))
        raw_shift_y = np.random.uniform(-0.5, 0.5, size=(num_frames, 1))

        min_x = centroids[:, 0:1] - (0.1 * scale)
        max_x = centroids[:, 0:1] + (0.1 * scale)
        min_y = centroids[:, 1:2] - (0.1 * scale)
        max_y = centroids[:, 1:2] + (0.1 * scale)

        allowed_min_dx = np.maximum(-0.5, -min_x)
        allowed_max_dx = np.minimum(0.5, 1.0 - max_x)
        allowed_min_dy = np.maximum(-0.5, -min_y)
        allowed_max_dy = np.minimum(0.5, 1.0 - max_y)

        valid_x = allowed_min_dx < allowed_max_dx
        allowed_max_dx = np.where(valid_x, allowed_max_dx, allowed_min_dx)

        valid_y = allowed_min_dy < allowed_max_dy
        allowed_max_dy = np.where(valid_y, allowed_max_dy, allowed_min_dy)

        shift_x = allowed_min_dx + np.random.rand(num_frames, 1) * (
            allowed_max_dx - allowed_min_dx
        )
        shift_y = allowed_min_dy + np.random.rand(num_frames, 1) * (
            allowed_max_dy - allowed_min_dy
        )

        shift_x = np.where(valid_x, shift_x, 0.0)
        shift_y = np.where(valid_y, shift_y, 0.0)

        aug_cx = centroids[:, 0:1] + shift_x
        aug_cy = centroids[:, 1:2] + shift_y

        cx = centroids[:, 0]
        cy = centroids[:, 1]
        aug_cx = aug_cx.flatten()
        aug_cy = aug_cy.flatten()

        # Pre-calculate exactly what plt.hist2d was doing!
        # bins=200 ensures we don't lose any resolution at all.
        h1_counts, xedges, yedges = np.histogram2d(
            cx, cy, bins=200, range=[[0, 1], [0, 1]]
        )
        h2_counts, _, _ = np.histogram2d(
            aug_cx, aug_cy, bins=200, range=[[0, 1], [0, 1]]
        )

        print(f"Caching aggregated heatmaps to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "h1_counts": h1_counts.tolist(),
                    "h2_counts": h2_counts.tolist(),
                    "xedges": xedges.tolist(),
                    "yedges": yedges.tolist(),
                },
                f,
            )

    t1 = time.time()
    print(f"Data ready in {t1 - t0:.1f}s")

    print("Plotting subplots with pcolormesh...")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.9, 2.5))

    # We must mask zeros to match hist2d cmin=1 behavior (white background)
    h1_counts_masked = np.ma.masked_where(h1_counts < 1, h1_counts)
    h2_counts_masked = np.ma.masked_where(h2_counts < 1, h2_counts)

    X, Y = np.meshgrid(xedges, yedges)

    # Note: histogram2d returns (x,y) counts, but pcolormesh expects (y,x) orientation, so we transpose.
    mesh1 = ax1.pcolormesh(X, Y, h1_counts_masked.T, cmap="plasma", vmin=1)
    ax1.invert_yaxis()
    ax1.set_facecolor("white")
    ax1.set_xlabel("Horizontal Position [normalized]")
    ax1.set_ylabel("Vertical Position [normalized]")

    # Plot Augmented (forced to use the same max color scale as the original)
    mesh2 = ax2.pcolormesh(
        X, Y, h2_counts_masked.T, cmap="plasma", vmin=1, vmax=h1_counts_masked.max()
    )
    ax2.invert_yaxis()
    ax2.set_facecolor("white")
    ax2.set_xlabel("Horizontal Position [normalized]")

    # Shared colorbar
    cbar = fig.colorbar(mesh1, ax=[ax1, ax2], fraction=0.05, pad=0.04)
    cbar.set_label("Frequency [frames]", labelpad=10)

    out_dir = "./MA/latex/images"
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = os.path.join(out_dir, "jester_spatial_heatmap_4x_bounded_v2.png")
    out_path_pgf = os.path.join(out_dir, "jester_spatial_heatmap_4x_bounded_v2.pgf")

    plt.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_path_pgf, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {out_path_png} and .pgf")


if __name__ == "__main__":
    plot_both()
