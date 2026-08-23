import re

import matplotlib.pyplot as plt


def parse_tegrastats(filepath):
    cpu_power = []
    gpu_power = []
    total_power = []
    sys5v_power = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        # Extract VDD_CPU_CV (current power, first value before slash)
        cpu_match = re.search(r"VDD_CPU_CV (\d+)mW", line)
        if cpu_match:
            cpu_power.append(int(cpu_match.group(1)))

        # Extract VDD_GPU_SOC
        gpu_match = re.search(r"VDD_GPU_SOC (\d+)mW", line)
        if gpu_match:
            gpu_power.append(int(gpu_match.group(1)))

        # Extract VIN_SYS_5V0
        sys_match = re.search(r"VIN_SYS_5V0 (\d+)mW", line)
        if sys_match:
            sys5v_power.append(int(sys_match.group(1)))

    # The true total "ALL" power is the sum of the main rails
    for i in range(len(cpu_power)):
        total = cpu_power[i] + gpu_power[i] + sys5v_power[i]
        total_power.append(total)

    return cpu_power, gpu_power, total_power, sys5v_power


import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "power_usage.txt")
cpu, gpu, total, sys5v = parse_tegrastats(log_file)

# We plot all seconds in the data
cpu = [x / 1000.0 for x in cpu]  # Convert to Watts
gpu = [x / 1000.0 for x in gpu]
total = [x / 1000.0 for x in total]
sys5v = [x / 1000.0 for x in sys5v]
time_axis = list(range(len(cpu)))

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

# Plot Total System Power and CPU Power
plt.plot(
    time_axis,
    total,
    label="Total Jetson Power",
    color="darkorange",
    linewidth=2.0,
    linestyle="-",
)
plt.plot(time_axis, cpu, label="CPU Power", color="steelblue", linewidth=2.0)

# Fill the CPU area to emphasize the compute footprint
plt.fill_between(time_axis, 0, cpu, color="steelblue", alpha=0.15)

# Formatting
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.ylim(0, 8.0)  # Adjusted for total power (can reach ~12W)
plt.xlim(0, 86)
plt.xticks(range(0, 86, 15))
plt.grid(True, linestyle="--", alpha=0.3)
plt.axvline(
    30, color="dimgray", linestyle=":", linewidth=1.5, alpha=0.8, label="Phase Change"
)
plt.axvline(60, color="dimgray", linestyle=":", linewidth=1.5, alpha=0.8)

plt.legend(loc="upper right")

out_dir = os.path.join(script_dir, "../../MA/latex/images")
os.makedirs(out_dir, exist_ok=True)

plot_path_png = os.path.join(out_dir, "power_usage_plot.png")
plot_path_pgf = os.path.join(out_dir, "power_usage_plot.pgf")

plt.savefig(plot_path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(plot_path_pgf, bbox_inches="tight", pad_inches=0.02)
print(f"Successfully saved {plot_path_png} and .pgf")
