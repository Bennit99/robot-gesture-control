import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_size():
    t0 = time.time()
    cache_path = "jester/plot_cache/size_dist.json"

    # Bins for distance in meters (0 to 3.0 meters)
    bins = np.linspace(0, 3.0, 150)

    if os.path.exists(cache_path):
        print("Loading cached aggregated distance histograms...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        hist1 = np.array(data["hist1"])
        hist2 = np.array(data["hist2"])
    else:
        print("Parsing parquets to compute hand sizes...")
        base_dir = "./data/jester/landmarks"
        size_list = []
        x_cols = [f"x{i}" for i in range(21)]
        y_cols = [f"y{i}" for i in range(21)]

        count = 0
        for root, _, files in os.walk(base_dir):
            if "landmarks.parquet" in files:
                path = os.path.join(root, "landmarks.parquet")
                try:
                    df = pd.read_parquet(path)

                    min_x = df[x_cols].min(axis=1).values
                    max_x = df[x_cols].max(axis=1).values
                    min_y = df[y_cols].min(axis=1).values
                    max_y = df[y_cols].max(axis=1).values

                    width = max_x - min_x
                    height = max_y - min_y
                    diag = np.sqrt(width**2 + height**2)

                    size_list.append(diag)
                    count += 1
                except Exception:
                    pass

        sizes = np.concatenate(size_list).astype(np.float32)

        # Avoid division by zero for invalid/tiny boxes
        valid = sizes > 0.05
        sizes = sizes[valid]
        num_frames = len(sizes)

        # Convert scale (s) to estimated distance in meters
        distances = 0.2 / sizes

        print("Simulating augmentation...")
        np.random.seed(42)
        dist_factor = np.random.uniform(1.0, 4.0, size=num_frames)
        aug_distances = distances * dist_factor

        # Pre-calculate the exact histogram bins
        hist1, _ = np.histogram(distances, bins=bins)
        hist2, _ = np.histogram(aug_distances, bins=bins)

        print(f"Caching aggregated histograms to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump({"hist1": hist1.tolist(), "hist2": hist2.tolist()}, f)

    t1 = time.time()
    print(f"Data ready in {t1 - t0:.1f}s")

    print("Plotting histograms...")
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

    max_freq = max(hist1.max(), hist2.max()) * 1.1

    # Plot Original
    ax1.stairs(hist1, bins, fill=True, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Estimated Distance [m]")
    ax1.set_ylabel("Frequency [frames]")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
    ax1.set_xlim(0, 3.0)
    ax1.set_ylim(0, max_freq)

    # Plot Augmented
    ax2.stairs(hist2, bins, fill=True, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Estimated Distance [m]")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
    ax2.set_xlim(0, 3.0)
    ax2.set_ylim(0, max_freq)

    out_dir = "./MA/latex/images"
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()

    out_path_png = os.path.join(out_dir, "jester_distance_distribution_4x.png")
    out_path_pgf = os.path.join(out_dir, "jester_distance_distribution_4x.pgf")

    plt.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_path_pgf, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {out_path_png} and .pgf")


if __name__ == "__main__":
    plot_size()
