import os
import statistics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cfg import cfg
from web_ui.streaming_web_server.backend import stats_ready
from utils.mediapipe_extensions import is_landmark_model
from utils.timeTracker import average_times


def _save_fps_stats(fps_values: list[float]) -> None:
    """Saves FPS statistics to the output directory."""
    if not fps_values:
        print("No FPS data to save.")
        return

    avg_fps = statistics.mean(fps_values)
    median_fps = statistics.median(fps_values)
    min_fps = min(fps_values)
    max_fps = max(fps_values)
    stdev_fps = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0
    total_frames = len(fps_values)

    stats_text = (
        f"FPS Statistics (LIVE_STREAM)\n"
        f"{'=' * 35}\n"
        f"Total Frames:  {total_frames}\n"
        f"Average FPS:   {avg_fps:.2f}\n"
        f"Median FPS:    {median_fps:.2f}\n"
        f"Min FPS:       {min_fps:.2f}\n"
        f"Max FPS:       {max_fps:.2f}\n"
        f"Std Dev:       {stdev_fps:.2f}\n"
    )

    print(stats_text)

    # Save stats text file
    stats_path = os.path.join(cfg.output, "fps_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"FPS stats saved to {stats_path}")

    # Save raw FPS values as CSV
    csv_path = os.path.join(cfg.output, "fps_values.csv")
    with open(csv_path, "w") as f:
        f.write("frame,fps\n")
        f.writelines(f"{i},{fps_val:.4f}\n" for i, fps_val in enumerate(fps_values))
    print(f"Raw FPS values saved to {csv_path}")


def _plot_fps_distribution(fps_values: list[float]) -> None:
    """Generates and saves an FPS distribution histogram to the output directory."""
    if not fps_values:
        print("No FPS data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Re-enable tick labels (may be globally disabled by plot.py rcParams)
    for ax in axes:
        ax.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)

    # Left plot: FPS distribution histogram
    ax1 = axes[0]
    ax1.hist(fps_values, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(
        statistics.mean(fps_values),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {statistics.mean(fps_values):.1f}",
    )
    ax1.axvline(
        statistics.median(fps_values),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {statistics.median(fps_values):.1f}",
    )
    ax1.set_xlabel("FPS")
    ax1.set_ylabel("Frequency")
    ax1.set_title("FPS Distribution (LIVE_STREAM)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right plot: FPS over time
    ax2 = axes[1]
    ax2.plot(fps_values, color="steelblue", linewidth=0.5, alpha=0.7)
    ax2.axhline(
        statistics.mean(fps_values),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {statistics.mean(fps_values):.1f}",
    )
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("FPS")
    ax2.set_title("FPS over Time (LIVE_STREAM)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(cfg.output, "fps_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"FPS distribution plot saved to {plot_path}")


def _save_and_plot_latency_stats() -> None:
    """Saves and plots latency statistics from timeTracker to the output directory.

    Generates a horizontal stacked bar chart showing the breakdown of time spent
    in each processing step for the average iteration.
    """
    avg_latency = average_times()
    if not avg_latency:
        print("No timeTracker data to save/plot.")
        return

    # Total latency across all steps
    total_avg = sum(avg_latency.values())

    # --- Save Stats Text ---
    stats_text = f"Pipeline Latency Breakdown (Averages)\n{'=' * 40}\n"
    for name, t in avg_latency.items():
        stats_text += f"{name.ljust(20)}: {t:.2f} ms ({(t / total_avg) * 100 if total_avg else 0:.1f}%)\n"
    stats_text += f"{'-' * 40}\n"
    stats_text += f"Total Average      : {total_avg:.2f} ms\n"

    print(stats_text)
    stats_path = os.path.join(cfg.output, "latency_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"Latency stats saved to {stats_path}")

    # --- Save Raw CSV Data (per step averages) ---
    csv_path = os.path.join(cfg.output, "latency_averages.csv")
    with open(csv_path, "w") as f:
        f.write("step_name,average_ms\n")
        for name, t in avg_latency.items():
            f.write(f"{name},{t:.4f}\n")
    print(f"Latency averages saved to {csv_path}")

    # --- Plot Horizontal Stacked Bar (Gantt-style) ---
    fig, ax = plt.subplots(figsize=(10, 1))

    names = list(avg_latency.keys())
    times = list(avg_latency.values())

    # Generate distinct colors for each segment
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    left = 0.0
    for idx, (name, t) in enumerate(zip(names, times)):
        pct = (t / total_avg) * 100 if total_avg else 0
        ax.barh(
            0,
            t,
            left=left,
            color=colors[idx],
            edgecolor="white",
            height=0.5,
            label=f"{name} ({t:.1f}ms)",
        )

        # Add percentage text inside the bar if it's wide enough
        if pct > 5:
            ax.text(
                left + t / 2,
                0,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=10,
            )

        left += t

    ax.set_yticks([])  # Remove y-axis ticks
    ax.tick_params(axis="x", bottom=True, labelbottom=True)

    ax.set_xlim(0, total_avg)
    interval = max(1, int(total_avg / 5))
    ticks = list(range(0, int(total_avg), interval))
    if ticks and (total_avg - ticks[-1]) < (interval * 0.5):
        ticks.pop()
    ticks.append(total_avg)

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{t}" if t != total_avg else f"Total: {t:.1f} ms" for t in ticks]
    )

    ax.set_xlabel("Time (ms)")
    ax.set_title("Average Processing Time per Frame")

    # Legend below the plot, items below each other
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        fancybox=True,
        shadow=False,
        ncol=1,
    )

    plt.tight_layout()
    plot_path = os.path.join(cfg.output, "latency_breakdown.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency breakdown plot saved to {plot_path}")
    
    _plot_latency_over_time()


def _plot_latency_over_time() -> None:
    """Generates and saves a total latency over time graph to the output directory."""
    from utils.timeTracker import iterations, names
    if not iterations or not names:
        return
        
    # Find indices to exclude (idle time / waiting for camera)
    exclude_indices = {i for i, name in enumerate(names) if name in ["idle / wait", "frontend_communication"]}
    
    # Calculate total processing latency per frame in ms (excluding idle time)
    total_latencies_ms = []
    for it in iterations:
        if len(it) > 0:
            processing_time = sum(val for i, val in enumerate(it) if i not in exclude_indices)
            total_latencies_ms.append(processing_time * 1000)
    
    if not total_latencies_ms:
        return
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
    
    ax.plot(total_latencies_ms, color="darkorange", linewidth=1.0, alpha=0.8)
    
    avg_latency = statistics.mean(total_latencies_ms) if total_latencies_ms else 0
    ax.axhline(
        avg_latency,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {avg_latency:.1f} ms",
    )
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("End-to-End Latency over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(cfg.output, "latency_over_time.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Latency over time plot saved to {plot_path}")


def _save_landmarks(results, output_dir: str = None):
    """Save hand landmark results as Parquet files in the output directory.

    Saves two files:
    - landmarks.parquet: normalized image landmarks (x, y, z per landmark per hand per frame)
    - world_landmarks.parquet: real-world 3D landmarks in meters

    Parquet is ~5-10x smaller than CSV and much faster to read.
    """
    output_dir = output_dir or cfg.output
    os.makedirs(output_dir, exist_ok=True)

    col_names = ["frame", "hand_index", "handedness"]
    for i in range(21):
        col_names.extend([f"x{i}", f"y{i}", f"z{i}"])

    def _build_dataframe(results, use_world=False):
        rows = []
        for frame_idx, result in enumerate(results):
            lm_list = (
                result.hand_world_landmarks if use_world else result.hand_landmarks
            )
            for hand_idx, (handedness, landmarks) in enumerate(
                zip(result.handedness, lm_list)
            ):
                # Store handedness as binary: 0 = Left, 1 = Right
                hand_binary = 0 if handedness[0].category_name == "Left" else 1
                row = [frame_idx, hand_idx, hand_binary]
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                rows.append(row)
        return pd.DataFrame(rows, columns=col_names)

    landmarks_path = os.path.join(output_dir, "landmarks.parquet")
    df = _build_dataframe(results, use_world=False)
    df.to_parquet(landmarks_path, index=False)
    print(
        f"Landmarks saved to {landmarks_path} ({len(results)} frames, {os.path.getsize(landmarks_path) / 1024:.1f} KB)"
    )

    world_path = os.path.join(output_dir, "world_landmarks.parquet")
    df_world = _build_dataframe(results, use_world=True)
    df_world.to_parquet(world_path, index=False)
    print(
        f"World landmarks saved to {world_path} ({len(results)} frames, {os.path.getsize(world_path) / 1024:.1f} KB)"
    )

    # Return number of hands detected per frame for stats accumulation
    hands_per_frame = [len(r.handedness) for r in results]
    handedness_list = [
        0 if h[0].category_name == "Left" else 1 for r in results for h in r.handedness
    ]
    # Return wrist positions (landmark 0) for heatmap (normalized x, y)
    wrist_positions = []
    for r in results:
        for lm in r.hand_landmarks:
            wrist_positions.append((lm[0].x, lm[0].y))
    return hands_per_frame, handedness_list, wrist_positions


def save_video_landmarks(results, video_name: str, stats_accumulator: dict = None):
    """Save landmarks for a single video to a subfolder: <output>/<video_name>/.

    Called incrementally during jester processing so only one video's
    worth of results needs to be in memory at a time.
    If stats_accumulator is provided, accumulates stats for later aggregate plotting.
    """
    video_dir = os.path.join(cfg.output, video_name)
    hands_per_frame, handedness_list, wrist_positions = _save_landmarks(
        results, output_dir=video_dir
    )

    if stats_accumulator is not None:
        stats_accumulator["hands_per_frame"].extend(hands_per_frame)
        stats_accumulator["handedness"].extend(handedness_list)
        stats_accumulator["wrist_positions"].extend(wrist_positions)
        stats_accumulator["videos_processed"] += 1
        stats_accumulator["total_frames"] += len(results)


def create_stats_accumulator() -> dict:
    """Create a fresh stats accumulator for incremental landmark processing."""
    return {
        "hands_per_frame": [],
        "handedness": [],
        "wrist_positions": [],
        "videos_processed": 0,
        "total_frames": 0,
    }


class JesterTracker:
    """Tracks jester video boundaries, saves per-video landmarks incrementally,
    and generates aggregate statistics on finalize().

    Usage in main:
        tracker = JesterTracker()
        for ...: tracker.add_result(result, video_folder)
        tracker.finalize()
    """

    def __init__(self):
        self._current_video = None
        self._video_results = []
        self._videos_saved = 0
        self._stats_acc = create_stats_accumulator()

    def add_result(self, result, video_folder: str):
        """Add a single frame result. Saves and flushes when the video folder changes."""
        if self._current_video is not None and video_folder != self._current_video:
            self._flush()
        self._current_video = video_folder
        self._video_results.append(result)

    def finalize(self):
        """Save the last video and generate aggregate statistics."""
        if self._video_results:
            self._flush()
        print(f"Saved landmarks for {self._videos_saved} videos total.")
        save_landmark_aggregate_stats(self._stats_acc)

    def _flush(self):
        """Save current video results and reset buffer."""
        save_video_landmarks(self._video_results, self._current_video, self._stats_acc)
        self._videos_saved += 1
        self._video_results = []


def save_landmark_aggregate_stats(stats: dict, output_dir: str = None):
    """Generate and save aggregate statistics and plots from accumulated landmark data.

    Creates:
    - landmark_stats.txt: summary statistics
    - hand_detection_distribution.png: bar chart of 0/1/2 hands per frame
    - handedness_distribution.png: left vs right hand pie chart
    - hand_position_heatmap.png: 2D heatmap of wrist positions in frame
    """
    output_dir = output_dir or cfg.output
    os.makedirs(output_dir, exist_ok=True)

    hands_per_frame = stats["hands_per_frame"]
    handedness_list = stats["handedness"]
    wrist_positions = stats["wrist_positions"]
    total_frames = stats["total_frames"]
    videos_processed = stats["videos_processed"]

    # --- Hand detection counts ---
    n_0_hands = hands_per_frame.count(0)
    n_1_hand = hands_per_frame.count(1)
    n_2_hands = hands_per_frame.count(2)

    # --- Handedness counts (0 = Left, 1 = Right) ---
    n_left = handedness_list.count(0)
    n_right = handedness_list.count(1)

    # --- Stats text ---
    stats_text = (
        f"Landmark Aggregate Statistics\n"
        f"{'=' * 40}\n"
        f"Videos processed:    {videos_processed}\n"
        f"Total frames:        {total_frames}\n"
        f"\nHand Detection per Frame:\n"
        f"  No hands:          {n_0_hands} ({100 * n_0_hands / total_frames:.1f}%)\n"
        f"  1 hand:            {n_1_hand} ({100 * n_1_hand / total_frames:.1f}%)\n"
        f"  2 hands:           {n_2_hands} ({100 * n_2_hands / total_frames:.1f}%)\n"
        f"\nHandedness Distribution:\n"
        f"  Left:              {n_left} ({100 * n_left / (n_left + n_right):.1f}%)\n"
        f"  Right:             {n_right} ({100 * n_right / (n_left + n_right):.1f}%)\n"
        f"\nTotal hand detections: {n_left + n_right}\n"
    )
    print(stats_text)
    stats_path = os.path.join(output_dir, "landmark_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"Stats saved to {stats_path}")

    # --- Plot 1: Hand detection distribution (0/1/2 hands) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
    categories = ["No hands", "1 hand", "2 hands"]
    counts = [n_0_hands, n_1_hand, n_2_hands]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total_frames if total_frames > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_frames * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Number of Frames")
    ax.set_title(
        f"Hand Detection Distribution ({total_frames} frames, {videos_processed} videos)"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "hand_detection_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Hand detection distribution saved to {plot_path}")

    # --- Plot 2: Handedness distribution (left vs right) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
    if n_left + n_right > 0:
        ax.pie(
            [n_left, n_right],
            labels=[f"Left ({n_left})", f"Right ({n_right})"],
            colors=["#3498db", "#e67e22"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
    ax.set_title("Handedness Distribution")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "handedness_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Handedness distribution saved to {plot_path}")

    # --- Plot 3: Heatmap of hand position in frame (wrist landmark) ---
    if wrist_positions:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
        xs = [p[0] for p in wrist_positions]
        ys = [p[1] for p in wrist_positions]
        # 2D histogram as heatmap (normalized coords: x=0..1, y=0..1)
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, bins=50, range=[[0, 1], [0, 1]]
        )
        # Transpose so x is horizontal, y is vertical (and flip y for image coords)
        im = ax.imshow(
            heatmap.T,
            origin="upper",  # y=0 is top of image
            extent=[0, 1, 1, 0],
            aspect="auto",
            cmap="hot",
            interpolation="gaussian",
        )
        plt.colorbar(im, ax=ax, label="Detection count")
        ax.set_xlabel("X position (normalized)")
        ax.set_ylabel("Y position (normalized)")
        ax.set_title(
            f"Hand Position Heatmap (wrist, {len(wrist_positions)} detections)"
        )
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "hand_position_heatmap.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Hand position heatmap saved to {plot_path}")
    else:
        print("No wrist positions to plot heatmap.")


def save_results(results):
    """Unified entry point: saves landmarks or gesture stats depending on model type."""
    if is_landmark_model():
        # For LIVE_STREAM, collect results from callback storage
        if cfg.running_mode == "LIVE_STREAM":
            from engine.callback import landmark_results, landmark_results_lock

            with landmark_results_lock:
                results = [r for r, _ in landmark_results]
                landmark_results.clear()
        _save_landmarks(results)
    else:
        _calculate_gesture_stats(results)


def save_livestream_session_stats() -> None:
    """Generate/save LIVE_STREAM stats from backend accumulators.

    Use this for pipelines that use a landmark model but still want
    gesture/FPS/latency stats in the web frontend (e.g. stacked dynamic pipeline).
    """
    _calculate_gesture_stats([])


def _calculate_gesture_stats(results):
    # In LIVE_STREAM mode, save FPS stats and plot from backend data
    if cfg.running_mode == "LIVE_STREAM":
        from web_ui.streaming_web_server.backend import (
            livestream_stats,
            livestream_stats_lock,
        )

        with livestream_stats_lock:
            fps_values = list(livestream_stats["fps_values"])
        if fps_values:
            _save_fps_stats(fps_values)
            _plot_fps_distribution(fps_values)
        else:
            print("No FPS data collected during LIVE_STREAM.")
        # Deactivated user-requested: only keep FPS plot in frontend
        # _save_and_plot_latency_stats()

        stats_ready.set()  # Signal that stats are ready
        print("Stats images are now available in the frontend.")

    else:
        # For IMAGE and VIDEO modes, calculate and save stats from results
        # Save results to out dir as a plain text file
        with open(cfg.output + "//result.txt", "w") as f:
            total_scores = 0
            total_gestures = 0
            for idx, result in enumerate(results):
                f.write(f"Result {idx:03} =\n")
                f.write(f"  Gestures: {result.gestures}\n")
                f.write(f"  Handedness: {result.handedness}\n")
                f.write(f"  Hand Landmarks: {result.hand_landmarks}\n")
                f.write(f"  Hand World Landmarks: {result.hand_world_landmarks}\n")
                f.write("\n")

                # Calculate total scores and gesture count for averaging
                for gesture_list in result.gestures:
                    if gesture_list:
                        total_scores += gesture_list[0].score
                        total_gestures += 1

        # Calculate stats
        frames = len(results)
        avg_hands_per_result = total_gestures / frames if frames > 0 else 0
        avg_score = total_scores / total_gestures if total_gestures > 0 else 0

        # Calculate total gestures detected (excluding 'None')
        total_detected_gestures = sum(
            1
            for r in results
            for gesture_list in r.gestures
            if gesture_list and gesture_list[0].category_name != "None"
        )

        print(f"Total Frames: {frames}")
        print(f"Total Hands Detected: {total_gestures}")
        print(f"Total Gestures Detected (excluding 'None'): {total_detected_gestures}")
        print(f"Average Hands per Frame: {avg_hands_per_result:.2f}")
        print(f"Average Gesture Score: {avg_score:.2f}")

        # Calculate gesture distribution
        gesture_counts = {}
        for r in results:
            for gesture_list in r.gestures:
                if gesture_list:
                    gesture_name = gesture_list[0].category_name
                    gesture_counts[gesture_name] = (
                        gesture_counts.get(gesture_name, 0) + 1
                    )

        # Sort gestures alphabetically
        sorted_gestures = sorted(gesture_counts.items())

        print("Gesture Distribution:")
        for gesture, count in sorted_gestures:
            print(f"  {gesture}: {count}")

        # Save stats as file in output dir
        with open(cfg.output + "//stats.txt", "w") as f:
            f.write(f"Total Frames: {frames}\n")
            f.write(f"Total Hands Detected: {total_gestures}\n")
            f.write(
                f"Total Gestures Detected (excluding 'None'): {total_detected_gestures}\n"
            )
            f.write(f"Average Hands per Frame: {avg_hands_per_result:.2f}\n")
            f.write(f"Average Gesture Score: {avg_score:.2f}\n")
            f.write("Gesture Distribution:\n")
            for gesture, count in sorted_gestures:
                f.write(f"  {gesture}: {count}\n")
