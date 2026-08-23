"""Evaluate a trained gesture classifier and produce a confusion matrix.

Usage
-----
::

    # Evaluate from a checkpoint file
    uv run python -m jester.training.evaluate --ckpt ./jester_runs/stgcn-1/checkpoints/best.ckpt

    # Evaluate from a run directory (auto-selects checkpoints/best.ckpt)
    uv run python -m jester.training.evaluate --ckpt ./jester_runs/stgcn-1

    # Shorthand (auto-resolves to ./jester_runs/stgcn-1)
    uv run python -m jester.training.evaluate --ckpt stgcn-1

    # Re-evaluate all runs under ./jester_runs
    uv run python -m jester.training.evaluate

    # Save without showing
    uv run python -m jester.training.evaluate --ckpt ./jester_runs/stgcn-1 --no-show --output ./eval_results
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from jester.training.data import (
    DEFAULT_SEQ_LEN,
    JesterLandmarkDataModule,
)
from jester.training.module import GestureClassifier

RUNS_ROOT = "./jester_runs"


def _estimate_recurrent_mflops(
    core_model: torch.nn.Module, seq_len: int
) -> float | None:
    """Estimate MFLOPs for LSTM/GRU blocks when fvcore can't count them.

    Uses a multiply+add convention (2 FLOPs per MAC), which aligns better
    with common model-compute reporting in papers/tooling.
    """
    rnn = None
    gates = 0
    if hasattr(core_model, "lstm"):
        rnn = core_model.lstm
        gates = 4
    elif hasattr(core_model, "gru"):
        rnn = core_model.gru
        gates = 3

    if rnn is None:
        return None

    hidden_size = int(rnn.hidden_size)
    num_layers = int(rnn.num_layers)
    num_directions = 2 if bool(rnn.bidirectional) else 1

    # Layer input size for stacked bidirectional RNNs
    layer_input_size = int(rnn.input_size)
    total_flops = 0.0

    for _layer in range(num_layers):
        # Per direction, per timestep:
        # 2 (mul+add) * gates * H * (I + H)
        per_dir_per_t = 2.0 * gates * hidden_size * (layer_input_size + hidden_size)
        total_flops += float(seq_len) * float(num_directions) * per_dir_per_t
        layer_input_size = hidden_size * num_directions

    return total_flops / 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained model and plot confusion matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint file OR run/checkpoint directory. "
        "Can be omitted to evaluate all runs under ./jester_runs. "
        "Shorthand like 'stgcn-1' is also supported.",
    )
    p.add_argument(
        "--landmarks", default="./data/jester/landmarks", help="Landmark root dir"
    )
    p.add_argument("--labels", default="./data/jester/lables", help="Labels CSV dir")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--max-hands", type=int, default=1, choices=[1, 2])
    p.add_argument("--world", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--cache-dir",
        default="./data/jester/cache",
        help="Cache dir for preloaded .pt files",
    )
    p.add_argument(
        "--interpolate-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear interpolation of missing landmark gaps",
    )
    p.add_argument(
        "--interpolate-max-gap",
        type=int,
        default=4,
        help="Maximum consecutive missing frames to interpolate",
    )
    p.add_argument(
        "--preload",
        default="auto",
        const="auto",
        nargs="?",
        help="Preload dataset: 'auto', 'cuda', 'cpu', or omit to disable",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Directory to save confusion matrix and metrics "
        "(default: auto-generated timestamped dir under ./jester_runs/)",
    )
    p.add_argument("--no-show", action="store_true", help="Don't display the plot")
    p.add_argument(
        "--normalize",
        default="true",
        choices=["true", "pred", "all", "none"],
        help="Confusion matrix normalisation axis",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only show top-N most confused class pairs",
    )
    p.add_argument(
        "--min-landmarks",
        type=int,
        default=0,
        help="Minimum required detected landmarks per video",
    )
    p.add_argument(
        "--sabbarese-mode",
        action="store_true",
        help="Enable Sabbarese sliding window validation mode",
    )
    return p.parse_args()


def _resolve_ckpt_path(ckpt_or_dir: str) -> tuple[str, str]:
    """Resolve user input to a concrete checkpoint path and run directory.

    Returns
    -------
    ckpt_path : absolute/relative path to a checkpoint file
    run_dir   : inferred run directory (used for default eval output)
    """
    path = os.path.normpath(ckpt_or_dir)

    # Shorthand support: --ckpt stgcn-1 -> ./jester_runs/stgcn-1
    if not os.path.exists(path):
        shorthand_path = os.path.join(RUNS_ROOT, path)
        if os.path.exists(shorthand_path):
            path = shorthand_path

    if os.path.isdir(path):
        # Accept either run dir (.../stgcn-1) or checkpoints dir (.../checkpoints)
        if os.path.basename(path) == "checkpoints":
            ckpt_path = os.path.join(path, "best.ckpt")
            run_dir = os.path.dirname(path)
        else:
            ckpt_path = os.path.join(path, "checkpoints", "best.ckpt")
            run_dir = path

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Could not find best checkpoint at: {ckpt_path}. "
                "Pass a checkpoint file directly with --ckpt."
            )
        return ckpt_path, run_dir

    if os.path.isfile(path):
        ckpt_path = path
        parent = os.path.dirname(path)
        run_dir = (
            os.path.dirname(parent)
            if os.path.basename(parent) == "checkpoints"
            else parent
        )
        return ckpt_path, run_dir

    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_or_dir}")


def _discover_run_dirs(root: str = RUNS_ROOT) -> list[str]:
    """Return run directories containing checkpoints/best.ckpt."""
    if not os.path.isdir(root):
        return []

    run_dirs: list[str] = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        best_ckpt = os.path.join(run_dir, "checkpoints", "best.ckpt")
        if os.path.isdir(run_dir) and os.path.exists(best_ckpt):
            run_dirs.append(run_dir)
    return run_dirs


@torch.no_grad()
def collect_predictions(
    model: GestureClassifier,
    datamodule: JesterLandmarkDataModule,
    split: str = "val",
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Run inference on a split and collect all predictions + labels."""
    model.eval()
    model.to(device)

    loader = (
        datamodule.val_dataloader() if split == "val" else datamodule.train_dataloader()
    )

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_video_ids: list[str] = []
    all_detection_rates: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Evaluating ({split})", unit="batch"):
        x = batch["x"].to(device)
        y = batch["y"]
        video_ids = batch.get("video_id", [""] * x.size(0))

        logits = model(x)
        preds = logits.argmax(dim=-1).cpu().numpy()

        # Detection rate in the model input sequence (% frames with >=1 hand point)
        frame_has_hand = x.abs().sum(dim=-1) > 0
        detection_rate = frame_has_hand.float().mean(dim=1).cpu().numpy() * 100.0

        all_preds.append(preds)
        all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        all_video_ids.extend(list(video_ids))
        all_detection_rates.append(detection_rate)

    all_preds_concat = np.concatenate(all_preds)
    all_labels_concat = np.concatenate(all_labels)
    all_detection_rates_concat = np.concatenate(all_detection_rates)

    if len(set(all_video_ids)) < len(all_video_ids):
        print(
            f"  [Evaluate] Aggregating {len(all_video_ids)} chunks into {len(set(all_video_ids))} videos via majority voting...",
            flush=True,
        )
        unique_vids = []
        agg_preds = []
        agg_labels = []
        agg_rates = []

        vid_to_indices: dict[str, list[int]] = {}
        for i, vid in enumerate(all_video_ids):
            if vid not in vid_to_indices:
                vid_to_indices[vid] = []
            vid_to_indices[vid].append(i)

        for vid, indices in vid_to_indices.items():
            unique_vids.append(vid)
            agg_labels.append(all_labels_concat[indices[0]])
            agg_rates.append(np.mean(all_detection_rates_concat[indices]))

            preds_for_vid = all_preds_concat[indices]
            values, counts = np.unique(preds_for_vid, return_counts=True)
            mode_pred = values[np.argmax(counts)]
            agg_preds.append(mode_pred)

        return (
            np.array(agg_preds),
            np.array(agg_labels),
            unique_vids,
            np.array(agg_rates),
        )

    return (
        all_preds_concat,
        all_labels_concat,
        all_video_ids,
        all_detection_rates_concat,
    )


def save_per_video_correctness(
    *,
    video_ids: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    detection_rate_percent: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """Save per-video predictions and correctness labels to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    true_names = [class_names[int(idx)] for idx in labels]
    pred_names = [class_names[int(idx)] for idx in preds]
    correct = preds == labels

    per_video = pd.DataFrame(
        {
            "video_id": video_ids,
            "true_label_idx": labels.astype(int),
            "pred_label_idx": preds.astype(int),
            "true_label": true_names,
            "pred_label": pred_names,
            "is_correct": correct.astype(bool),
            "detection_rate_percent": detection_rate_percent.astype(float),
        }
    )

    path = os.path.join(output_dir, "per_video_predictions.csv")
    per_video.to_csv(path, index=False)
    print(f"  Saved per-video predictions  {path}")
    return per_video


def plot_detection_rate_correct_vs_wrong(
    per_video: pd.DataFrame,
    output_dir: str,
    split: str,
    show: bool = False,
) -> str | None:
    """Plot per-video detection-rate histogram split by correctness."""
    if per_video.empty:
        return None

    correct_rates = per_video.loc[
        per_video["is_correct"], "detection_rate_percent"
    ].to_numpy()
    wrong_rates = per_video.loc[
        ~per_video["is_correct"], "detection_rate_percent"
    ].to_numpy()

    if len(correct_rates) == 0 and len(wrong_rates) == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 40, 41)

    # Convert percentage to frame count (0-40)
    correct_rates = correct_rates / 100.0 * 40.0
    wrong_rates = wrong_rates / 100.0 * 40.0

    # Calculate absolute counts per bin
    correct_counts, _ = np.histogram(correct_rates, bins=bins)
    wrong_counts, _ = np.histogram(wrong_rates, bins=bins)

    total_counts = correct_counts + wrong_counts
    valid = total_counts > 0

    # Calculate relative percentages (avoid div by zero)
    correct_pct = np.zeros_like(correct_counts, dtype=float)
    wrong_pct = np.zeros_like(wrong_counts, dtype=float)

    correct_pct[valid] = correct_counts[valid] / total_counts[valid] * 100
    wrong_pct[valid] = wrong_counts[valid] / total_counts[valid] * 100

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.95  # Slight gap between bars

    ax.bar(
        bin_centers[valid],
        correct_pct[valid],
        width=width,
        label=f"Correct ({len(correct_rates):,} total)",
        color="#2ecc71",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar(
        bin_centers[valid],
        wrong_pct[valid],
        width=width,
        bottom=correct_pct[valid],
        label=f"Wrong ({len(wrong_rates):,} total)",
        color="#e74c3c",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Detected landmarks in model input sequence (frames)")
    ax.set_ylabel("Relative Proportion (%)")
    # ax.set_title(f"Per-video Detection Rate by Correctness ({split})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "per_video_detection_rate_correct_vs_wrong.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved  {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: str = "true",
    output_dir: str = "./eval_results",
    show: bool = True,
) -> str:
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm          : raw confusion matrix (n_classes x n_classes)
    class_names : list of class labels
    normalize   : 'true' (recall), 'pred' (precision), 'all', or 'none'
    output_dir  : save directory
    show        : whether to call plt.show()

    Returns the path to the saved image.
    """
    n = len(class_names)

    # Normalise
    if normalize == "true":
        # Rows sum to 1 (recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float)
        )
        cm_norm = np.nan_to_num(cm_norm)
        fmt = ".2f"
        title_suffix = "(Row-Normalized)"
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_norm = np.divide(
            cm, col_sums, where=col_sums != 0, out=np.zeros_like(cm, dtype=float)
        )
        fmt = ".1%"
        title_suffix = "(normalised by prediction)"
    elif normalize == "all":
        total = cm.sum()
        cm_norm = cm / total if total > 0 else cm.astype(float)
        fmt = ".2%"
        title_suffix = "(normalised over all)"
    else:
        cm_norm = cm.astype(float)
        fmt = "d"
        title_suffix = "(raw counts)"

    # Determine figure size based on class count
    size = max(10, n * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))

    # Heatmap
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    # Text annotations (skip if too many classes to be readable)
    if n <= 35:
        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                if val == 0:
                    continue
                text = f"{val:{fmt}}" if fmt != "d" else f"{int(cm[i, j])}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=5,
                )

    ax.set_xlabel("Predicted Gesture")
    ax.set_ylabel("Actual Gesture")
    # ax.set_title(f"Normalized Confusion Matrix" if "Row-Normalized" in title_suffix else f"Confusion Matrix {title_suffix}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200)
    print(f"  Saved  {path}")

    if show:
        plt.show()
    plt.close(fig)
    return path


def print_classification_report(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str = "./eval_results",
) -> str:
    """Print per-class precision, recall, F1 and save to file.

    Returns the report string.
    """
    n = len(class_names)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(
        tp, tp + fp, where=(tp + fp) > 0, out=np.zeros(n, dtype=float)
    )
    recall = np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.zeros(n, dtype=float))
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        where=(precision + recall) > 0,
        out=np.zeros(n, dtype=float),
    )

    # Overall
    total = support.sum()
    accuracy = tp.sum() / total if total > 0 else 0
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support) if total > 0 else 0

    # Format table
    header = f"{'Class':<35s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Support':>8s}"
    sep = "-" * len(header)
    lines = [header, sep]
    for i in range(n):
        lines.append(
            f"{class_names[i]:<35s} {precision[i]:>6.3f} {recall[i]:>6.3f} "
            f"{f1[i]:>6.3f} {int(support[i]):>8d}"
        )
    lines.append(sep)
    lines.append(
        f"{'Accuracy':<35s} {'':>6s} {'':>6s} {accuracy:>6.3f} {int(total):>8d}"
    )
    lines.append(
        f"{'Macro avg':<35s} {precision.mean():>6.3f} {recall.mean():>6.3f} "
        f"{macro_f1:>6.3f} {int(total):>8d}"
    )
    lines.append(
        f"{'Weighted avg':<35s} {np.average(precision, weights=support):>6.3f} "
        f"{np.average(recall, weights=support):>6.3f} "
        f"{weighted_f1:>6.3f} {int(total):>8d}"
    )

    report = "\n".join(lines)
    print(f"\n{report}\n")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved  {path}")
    return report


def print_most_confused(
    cm: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
) -> None:
    """Print the top-N most confused class pairs."""
    n = len(class_names)
    pairs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top-{top_n} most confused pairs (true  predicted : count):")
    for i, j, count in pairs[:top_n]:
        total_true = cm[i].sum()
        pct = 100 * count / total_true if total_true > 0 else 0
        print(
            f"    {class_names[i]:>30s}  {class_names[j]:<30s}  {count:>5d}  ({pct:.1f}%)"
        )


def evaluate(
    args: argparse.Namespace | None = None,
    *,
    run_info: dict | None = None,
) -> None:
    """Full evaluation pipeline.

    Parameters
    ----------
    args     : CLI namespace (parsed if None)
    run_info : optional metadata dict to save alongside results,
               linking them to a specific training run / checkpoint.
    """
    if args is None:
        args = parse_args()

    # Batch mode: no --ckpt => evaluate all run directories
    if not args.ckpt:
        run_dirs = _discover_run_dirs()
        if not run_dirs:
            raise FileNotFoundError(
                "No run directories with checkpoints/best.ckpt found under ./jester_runs"
            )

        print(f"\n  Found {len(run_dirs)} run(s) to evaluate under {RUNS_ROOT}")
        import copy

        for run_dir in run_dirs:
            sub_args = copy.copy(args)
            sub_args.ckpt = run_dir
            if args.output:
                sub_args.output = os.path.join(args.output, os.path.basename(run_dir))
            else:
                sub_args.output = None

            print("\n" + "=" * 60)
            print(f"  Evaluating run: {os.path.basename(run_dir)}")
            print("=" * 60)
            evaluate(sub_args, run_info=run_info)
        return

    # Resolve checkpoint path from file or directory input
    resolved_ckpt, run_dir = _resolve_ckpt_path(args.ckpt)
    args.ckpt = resolved_ckpt

    # -- Auto-generate output dir if not provided --------------------------
    if args.output is None:
        args.output = os.path.join(run_dir, "eval")

    # -- Load checkpoint ---------------------------------------------------
    print(f"\n  Loading checkpoint: {args.ckpt}")
    model = GestureClassifier.load_from_checkpoint(args.ckpt, map_location="cpu")
    print(f"  Architecture: {model.hparams.arch}")
    print(f"  Num classes:  {model.hparams.num_classes}")

    # -- Model stats: Parameters (M) and Compute (MFLOPs) ----------------
    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = n_params / 1e6
    mflops: float | None = None
    mflops_source = "unknown"
    fvcore_mflops: float | None = None
    recurrent_estimate_mflops: float | None = None
    print(f"  Parameters:   {n_params_m:.2f}M ({n_params:,})")
    # Compute MFLOPs (approximate, for a single forward pass)
    try:
        # Try to use fvcore if available
        try:
            from fvcore.nn import FlopCountAnalysis

            # Use a dummy input with batch size 1, seq_len, feature_dim
            if hasattr(model, "model") and hasattr(model.model, "forward"):
                input_dim = model.hparams.input_dim
                seq_len = model.hparams.seq_len
                dummy = torch.randn(1, seq_len, input_dim)
                model.model.eval()
                flop_analysis = FlopCountAnalysis(model.model, dummy)
                flops = flop_analysis.total()
                fvcore_mflops = float(flops) / 1e6

                unsupported = flop_analysis.unsupported_ops()
                has_recurrent_unsupported = (
                    unsupported.get("aten::lstm", 0) > 0
                    or unsupported.get("aten::gru", 0) > 0
                )

                if has_recurrent_unsupported:
                    recurrent_estimate_mflops = _estimate_recurrent_mflops(
                        model.model,
                        seq_len=seq_len,
                    )
                    if recurrent_estimate_mflops is not None:
                        mflops = recurrent_estimate_mflops
                        mflops_source = "recurrent_estimate"
                        print(
                            "  Compute:      "
                            f"{mflops:.2f} MFLOPs (estimated; fvcore undercounts recurrent ops)"
                        )
                        print(
                            "  Note:         "
                            f"fvcore raw={fvcore_mflops:.5f} MFLOPs, unsupported={dict(unsupported)}"
                        )
                    else:
                        mflops = fvcore_mflops
                        mflops_source = "fvcore_partial"
                        print(
                            "  Compute:      "
                            f"{mflops:.2f} MFLOPs (fvcore partial; unsupported ops present)"
                        )
                else:
                    mflops = fvcore_mflops
                    mflops_source = "fvcore"
                    print(
                        f"  Compute:      {mflops:.2f} MFLOPs (theoretical, 1x forward)"
                    )
            else:
                print("  Compute:      (model does not support fvcore flop analysis)")
                mflops_source = "unavailable"
        except ImportError:
            print("  Compute:      (Install fvcore for MFLOPs estimate)")
            mflops_source = "missing_fvcore"
    except Exception as e:
        print(f"  Compute:      (Error estimating MFLOPs: {e})")
        mflops_source = "error"

    # -- Device ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # -- DataModule --------------------------------------------------------
    preload_device = None
    if args.preload is not None:
        preload_device = (
            "cuda"
            if args.preload == "auto" and torch.cuda.is_available()
            else args.preload
        )
        if preload_device == "auto":
            preload_device = "cpu"

    dm = JesterLandmarkDataModule(
        landmarks_dir=args.landmarks,
        labels_dir=args.labels,
        seq_len=args.seq_len,
        max_hands=args.max_hands,
        world=args.world,
        interpolate_missing=args.interpolate_missing,
        interpolate_max_gap=args.interpolate_max_gap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preload_device=preload_device,
        min_landmarks=args.min_landmarks,
        sabbarese_mode=getattr(args, "sabbarese_mode", False),
    )
    dm.cache_dir = args.cache_dir if args.cache_dir else None
    dm.setup("fit")

    # -- Class names -------------------------------------------------------
    label_map = dm.label_map  # {name: idx}
    class_names = [""] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name

    # -- Collect predictions -----------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    # Main evaluation is always on validation split
    preds, labels, video_ids, detection_rate_percent = collect_predictions(
        model, dm, split="val", device=device
    )
    n_classes = len(class_names)

    print("\n  Split: val")
    print(f"  Samples: {len(preds)}")
    accuracy = (preds == labels).mean()
    print(f"  Accuracy: {accuracy:.4f}")

    # -- Build confusion matrix --------------------------------------------
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    # -- Report -----------------------------------------------------------
    print_classification_report(cm, class_names, args.output)
    top_n = args.top_n or min(15, n_classes)
    print_most_confused(cm, class_names, top_n=top_n)

    # -- Plot -------------------------------------------------------------
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=args.normalize,
        output_dir=args.output,
        show=not args.no_show,
    )

    # Save val-only per-video predictions
    per_video_val = save_per_video_correctness(
        video_ids=video_ids,
        labels=labels,
        preds=preds,
        class_names=class_names,
        detection_rate_percent=detection_rate_percent,
        output_dir=args.output,
    )
    per_video_val["split"] = "val"

    # Exclude "No gesture" class from correctness-vs-detection-rate plot
    per_video_plot = per_video_val[
        per_video_val["true_label"].str.casefold() != "no gesture"
    ].copy()
    dropped = len(per_video_val) - len(per_video_plot)
    if dropped > 0:
        print(f"  Dropped {dropped:,} 'No gesture' samples from detection-rate plot")

    plot_detection_rate_correct_vs_wrong(
        per_video_plot,
        output_dir=args.output,
        split="val",
        show=not args.no_show,
    )

    # -- Save raw matrix --------------------------------------------------
    np.save(os.path.join(args.output, "confusion_matrix.npy"), cm)
    print(f"  Saved raw matrix  {os.path.join(args.output, 'confusion_matrix.npy')}")

    all_summaries: dict[str, dict] = {
        "val": {
            "accuracy": float(accuracy),
            "num_samples": len(preds),
            "num_classes": n_classes,
            "output_dir": os.path.abspath(args.output),
        },
    }

    # -- Save run summary (links results to model/run) -----------------
    summary: dict = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": os.path.abspath(args.ckpt),
        "split": "val",
        "splits": all_summaries,
        "model_stats": {
            "parameters": int(n_params),
            "parameters_m": float(n_params_m),
            "mflops": (float(mflops) if mflops is not None else None),
            "mflops_source": mflops_source,
            "mflops_fvcore_raw": (
                float(fvcore_mflops) if fvcore_mflops is not None else None
            ),
            "mflops_recurrent_estimate": (
                float(recurrent_estimate_mflops)
                if recurrent_estimate_mflops is not None
                else None
            ),
        },
    }
    if run_info:
        summary["run"] = run_info

    summary_path = os.path.join(args.output, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary  {summary_path}")

    # -- Optional W&B logging ---------------------------------------------
    if run_info and run_info.get("wandb_project") and run_info.get("wandb_run_id"):
        try:
            wb = wandb.init(
                project=run_info["wandb_project"],
                id=run_info["wandb_run_id"],
                resume="allow",
            )
            log_payload: dict[str, float | int | None] = {
                "model/parameters": int(n_params),
                "model/parameters_m": float(n_params_m),
                "model/mflops": (float(mflops) if mflops is not None else None),
            }
            log_payload["eval/val_accuracy"] = float(all_summaries["val"]["accuracy"])
            wb.log(log_payload)
            wb.finish()
            print("  Logged model stats + eval metrics to W&B")
        except Exception as exc:
            print(f"  W&B logging skipped: {exc}")

    print("\nDone.\n")


if __name__ == "__main__":
    evaluate()
