import re

import matplotlib.pyplot as plt


def parse_cpu_stats(filepath):
    avg_usages = []
    max_freqs = []
    avg_freqs = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        # Extract the CPU block: CPU [23%@729,21%@729,...]
        cpu_match = re.search(r"CPU \[([^\]]+)\]", line)
        if cpu_match:
            cores_data = cpu_match.group(1).split(",")

            usages = []
            freqs = []

            for core in cores_data:
                # Some cores might be 'off' or in a different state, but usually format is XX%@YYYY
                parts = core.split("%@")
                if len(parts) == 2:
                    usages.append(int(parts[0]))
                    freqs.append(int(parts[1]))

            if usages and freqs:
                avg_usages.append(sum(usages) / len(usages))
                max_freqs.append(max(freqs))
                avg_freqs.append(sum(freqs) / len(freqs))

    return avg_usages, max_freqs, avg_freqs


import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "power_usage.txt")
avg_usages, max_freqs, avg_freqs = parse_cpu_stats(log_file)
time_axis = list(range(len(avg_usages)))

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

fig, ax1 = plt.subplots(figsize=(5.9, 2.5))

# Plot CPU Usage on the primary y-axis
color = "tab:blue"
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Avg CPU Utilization (%)", color=color)
ax1.plot(time_axis, avg_usages, color=color, linewidth=2, label="Avg Utilization")
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0, 100)
ax1.set_xlim(0, 86)
ax1.set_xticks(range(0, 86, 15))
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.axvline(
    30, color="dimgray", linestyle=":", linewidth=1.5, alpha=0.8, label="Phase Change"
)
ax1.axvline(60, color="dimgray", linestyle=":", linewidth=1.5, alpha=0.8)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = "tab:red"
# We plot the average frequency across cores
ax2.set_ylabel("Average Clock Speed (MHz)", color=color)
ax2.plot(
    time_axis, avg_freqs, color=color, linewidth=2, linestyle="-", label="Avg Core Freq"
)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(729, 2200)  # AGX Orin CPU idles at 729 MHz, maxes around ~2.2 GHz

out_dir = os.path.join(script_dir, "../../MA/latex/images")
os.makedirs(out_dir, exist_ok=True)

plot_path_png = os.path.join(out_dir, "cpu_stats_plot.png")
plot_path_pgf = os.path.join(out_dir, "cpu_stats_plot.pgf")

plt.savefig(plot_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(plot_path_pgf, bbox_inches="tight", pad_inches=0.02)
print(f"Successfully saved {plot_path_png} and .pgf")
