import json
import os

import matplotlib.pyplot as plt

# Configure plot styling
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.figsize": (5.9, 3.0),
    }
)


def main():
    arch_colors = {
        "conv1d": "#1f77b4",  # blue
        "lstm": "#ff7f0e",  # orange
        "gru": "#2ca02c",  # green
        "stgcn": "#d62728",  # red
        "transformer": "#9467bd",  # purple
    }

    arch_labels = {
        "conv1d": "Conv1D",
        "lstm": "LSTM",
        "gru": "GRU",
        "stgcn": "ST-GCN",
        "transformer": "Transformer",
    }

    cache_path = "jester/plot_cache/flop_scaling_data.json"
    if not os.path.exists(cache_path):
        print("Cache not found!")
        return

    with open(cache_path, "r") as f:
        data = json.load(f)

    plt.figure()

    for arch, arch_data in data.items():
        mflops = arch_data["mflops"][:7]
        accs = arch_data["accuracy"][:7]
        lbl = arch_labels.get(arch, arch)
        color = arch_colors.get(arch, "black")

        plt.plot(
            mflops, accs, marker="o", linewidth=2, markersize=6, label=lbl, color=color
        )

        if arch == "gru":
            # Highlight the 3rd point (Selected Model)
            plt.annotate(
                "Selected\nModel",
                xy=(mflops[2], accs[2]),
                xytext=(mflops[2] * 0.6, accs[2] + 0.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                fontsize=10,
                ha="right",
                va="bottom",
            )

            # Highlight the point with highest accuracy (index 5)
            plt.annotate(
                "Alternative\nOption",
                xy=(mflops[5], accs[5]),
                xytext=(mflops[5] * 1.5, accs[5] + 1.1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontsize=10,
                ha="left",
                va="top",
                color="gray",
            )

    plt.xscale("log")
    plt.xlabel("Computational Cost (MFLOPs)")
    plt.ylabel("Validation Accuracy (%)")

    plt.ylim(84, 93)

    from matplotlib import ticker

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.legend(title="Architecture", loc="lower right")
    plt.tight_layout()

    out_path = "MA/latex/images/flop_scaling_accuracy.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)

    out_path_png = "MA/latex/images/flop_scaling_accuracy.png"
    plt.savefig(
        out_path_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02
    )

    out_path_pgf = "MA/latex/images/flop_scaling_accuracy.pgf"
    plt.savefig(out_path_pgf, format="pgf", bbox_inches="tight", pad_inches=0.02)

    print(f"Saved plots to {out_path}, .png, and .pgf")


if __name__ == "__main__":
    main()
