import json
import os

import matplotlib.pyplot as plt

json_path = "output/ablation_study/interpolation_ablation_results.json"
if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    print("Please run scripts/evaluate_landmark_interpolation.py first.")
    exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

gaps = [r["Gap"] for r in data["results"]]
accuracies = [r["Accuracy"] for r in data["results"]]

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

plt.figure(figsize=(5.9, 2.5))
plt.plot(gaps, accuracies, marker="o", linestyle="-", color="b")

plt.xlabel("Maximum Interpolation Gap (Frames)")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.7)

# Set x-ticks explicitly so it shows integers
plt.xticks(gaps)

out_dir = "MA/latex/images"
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "interpolation_accuracy.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.savefig(
    os.path.join(out_dir, "interpolation_accuracy.pgf"),
    bbox_inches="tight",
    pad_inches=0.02,
)
print("Plot saved to .png and .pgf")
