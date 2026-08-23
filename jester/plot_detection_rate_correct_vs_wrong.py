import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_and_plot():
    cache_path = "jester/plot_cache/detection_rate_correct_vs_wrong.json"
    csv_path = "jester_runs/gru-9/eval/per_video_predictions.csv"

    bins = np.linspace(0, 40, 41)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        correct_pct = np.array(data["correct_pct"])
        wrong_pct = np.array(data["wrong_pct"])
        valid = np.array(data["valid"], dtype=bool)
        total_correct = data["total_correct"]
        total_wrong = data["total_wrong"]
    else:
        df = pd.read_csv(csv_path)
        # Filter out "No gesture"
        df = df[df["true_label"] != "No gesture"]

        correct_rates = (
            df.loc[df["is_correct"], "detection_rate_percent"].to_numpy() / 100.0 * 40.0
        )
        wrong_rates = (
            df.loc[~df["is_correct"], "detection_rate_percent"].to_numpy()
            / 100.0
            * 40.0
        )

        total_correct = len(correct_rates)
        total_wrong = len(wrong_rates)

        correct_counts, _ = np.histogram(correct_rates, bins=bins)
        wrong_counts, _ = np.histogram(wrong_rates, bins=bins)

        total_counts = correct_counts + wrong_counts
        valid = total_counts > 0

        correct_pct = np.zeros_like(correct_counts, dtype=float)
        wrong_pct = np.zeros_like(wrong_counts, dtype=float)

        correct_pct[valid] = correct_counts[valid] / total_counts[valid] * 100
        wrong_pct[valid] = wrong_counts[valid] / total_counts[valid] * 100

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "correct_pct": correct_pct.tolist(),
                    "wrong_pct": wrong_pct.tolist(),
                    "valid": valid.tolist(),
                    "total_correct": total_correct,
                    "total_wrong": total_wrong,
                },
                f,
                indent=2,
            )

    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.95

    ax.bar(
        bin_centers[valid],
        correct_pct[valid],
        width=width,
        label="Correct",
        color="#2ecc71",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar(
        bin_centers[valid],
        wrong_pct[valid],
        width=width,
        bottom=correct_pct[valid],
        label="Wrong",
        color="#e74c3c",
        edgecolor="black",
        alpha=0.8,
    )

    # Restore slight border around x-axis
    ax.set_xlim(-1, 41)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Detected landmarks in model input sequence (frames)")
    ax.set_ylabel("Relative Proportion (%)")
    # ax.set_title("Per-video Detection Rate by Correctness")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    # Save the output
    out_path_png = "MA/latex/images/per_video_detection_rate_correct_vs_wrong.png"
    out_path_pgf = "MA/latex/images/per_video_detection_rate_correct_vs_wrong.pgf"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_path_pgf, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved plot to {out_path_png} and .pgf")
    print(f"Cached stats to {cache_path}")


if __name__ == "__main__":
    generate_and_plot()
