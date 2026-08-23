"""Training entry point for Jester gesture classification.

Usage
-----
::

    # Default: LSTM, wandb logging, cosine scheduler
    uv run python -m jester.training.train

    # Transformer architecture, 100 epochs
    uv run python -m jester.training.train --arch transformer --epochs 100

    # Sweep: try GRU with larger hidden dim
    uv run python -m jester.training.train --arch gru --hidden_dim 256 --num_layers 3

    # Disable wandb (offline run)
    uv run python -m jester.training.train --no-wandb

All hyperparameters are logged to W&B automatically.
"""

from __future__ import annotations

import argparse
import os

import torch

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from jester.training.architectures import ARCHITECTURE_REGISTRY
from jester.training.data import JesterLandmarkDataModule
from jester.training.evaluate import evaluate
from jester.training.module import GestureClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a gesture classifier on Jester landmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data --------------------------------------------------------------
    g = p.add_argument_group("Data")
    g.add_argument(
        "--landmarks", default="./data/jester/landmarks", help="Landmark root dir"
    )
    g.add_argument("--labels", default="./data/jester/lables", help="Labels CSV dir")
    g.add_argument(
        "--seq-len", type=int, default=40, help="Fixed sequence length (frames)"
    )
    g.add_argument(
        "--max-hands", type=int, default=1, choices=[1, 2], help="Hands per frame"
    )
    g.add_argument("--world", action="store_true", help="Use world landmarks")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument(
        "--preload",
        default="auto",
        const="auto",
        nargs="?",
        help="Preload entire dataset into memory/VRAM. "
        "Values: 'auto' (GPU if available, else CPU), 'cuda', 'cpu'. "
        "None to load lazily from disk.",
    )
    g.add_argument(
        "--cache-dir",
        default="./data/jester/cache",
        help="Directory for consolidated .pt cache files. "
        "First run builds from Parquets; subsequent runs load instantly. "
        "Set to empty string to disable caching.",
    )
    g.add_argument(
        "--interpolate-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear interpolation of missing landmark gaps",
    )
    g.add_argument(
        "--interpolate-max-gap",
        type=int,
        default=4,
        help="Maximum consecutive missing frames to interpolate",
    )
    g.add_argument(
        "--min-landmarks",
        type=int,
        default=9,
        help="Drop videos with fewer than this many frames containing "
        "detected landmarks (0 = keep all)",
    )
    g.add_argument(
        "--augment",
        action="store_true",
        help="Apply random scaling and shifting augmentation to landmarks",
    )
    g.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[1.0, 2.0],
        help="Random isotropic scale range for augmentation",
    )
    g.add_argument(
        "--shift-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[-0.2, 0.2],
        help="Random spatial shift range (x/y) for augmentation",
    )
    g.add_argument(
        "--sabbarese-mode",
        action="store_true",
        help="Enable Sabbarese sliding window training and validation mode.",
    )

    # -- Model -------------------------------------------------------------
    g = p.add_argument_group("Model")
    g.add_argument(
        "--arch",
        default="lstm",
        choices=sorted(ARCHITECTURE_REGISTRY),
        help="Architecture to train",
    )
    g.add_argument(
        "--flop-scale",
        type=float,
        default=1.0,
        help="Scale factor for architecture FLOPs (multiplies channels by sqrt(scale))",
    )
    # Architecture-specific kwargs (passed through as **kwargs)
    g.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim (LSTM/GRU)")
    g.add_argument(
        "--num_layers", type=int, default=2, help="Num layers (LSTM/GRU/Transformer)"
    )
    g.add_argument("--d_model", type=int, default=128, help="Model dim (Transformer)")
    g.add_argument("--nhead", type=int, default=4, help="Attention heads (Transformer)")
    g.add_argument(
        "--dim_feedforward", type=int, default=256, help="FFN dim (Transformer)"
    )
    g.add_argument(
        "--unidirectional", action="store_true", help="Set bidirectional=False for RNNs"
    )
    g.add_argument(
        "--stgcn-channels",
        type=int,
        nargs=3,
        metavar=("C1", "C2", "C3"),
        default=[64, 64, 128],
        help="ST-GCN block channels (for --arch stgcn)",
    )
    g.add_argument(
        "--conv1d-channels",
        type=int,
        nargs=3,
        metavar=("C1", "C2", "C3"),
        default=[312, 624, 1248],
        help="Conv1D channels (for --arch conv1d)",
    )
    g.add_argument(
        "--stgcn-temporal-kernel-size",
        type=int,
        default=9,
        help="ST-GCN temporal kernel size (for --arch stgcn)",
    )
    g.add_argument("--dropout", type=float, default=0.3)

    # -- Training ----------------------------------------------------------
    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "plateau", "none"]
    )
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--label-smoothing", type=float, default=0.1)
    g.add_argument("--gradient-clip", type=float, default=1.0)
    g.add_argument("--early-stop-patience", type=int, default=10)
    g.add_argument("--precision", default="16-mixed", help="Training precision")
    g.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Path to checkpoint to resume training from (restores weights, "
        "optimizer, epoch, etc.)",
    )

    # -- Logging -----------------------------------------------------------
    g = p.add_argument_group("Logging")
    g.add_argument("--project", default="jester-gesture", help="W&B project name")
    g.add_argument(
        "--run-name", default=None, help="W&B run name (auto-generated if None)"
    )
    g.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    g.add_argument(
        "--wandb-run-id",
        default=None,
        help="W&B run ID to resume (use with --resume)",
    )
    g.add_argument(
        "--save-dir", default="./jester_runs", help="Checkpoint / log save dir"
    )
    g.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip automatic evaluation after training",
    )

    return p.parse_args()


def _build_arch_kwargs(args: argparse.Namespace) -> dict:
    """Extract architecture-specific kwargs from parsed args and scale by flop_scale."""
    import math

    scale_multiplier = math.sqrt(args.flop_scale)

    d_model = int(args.d_model * scale_multiplier)
    d_model = max(args.nhead, (d_model // args.nhead) * args.nhead)

    kwargs = {
        "hidden_dim": int(args.hidden_dim * scale_multiplier),
        "num_layers": args.num_layers,
        "d_model": d_model,
        "nhead": args.nhead,
        "dim_feedforward": int(args.dim_feedforward * scale_multiplier),
        "temporal_kernel_size": args.stgcn_temporal_kernel_size,
        "dropout": args.dropout,
    }

    if args.unidirectional:
        kwargs["bidirectional"] = False

    if args.arch == "stgcn":
        kwargs["channels"] = tuple(
            int(c * scale_multiplier) for c in args.stgcn_channels
        )
    elif args.arch == "conv1d":
        kwargs["channels"] = tuple(
            int(c * scale_multiplier) for c in args.conv1d_channels
        )

    return kwargs


def train(args: argparse.Namespace | None = None) -> None:
    """Run a full training loop."""
    if args is None:
        args = parse_args()

    # -- Device Info -------------------------------------------------------
    gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    print("\nPyTorch GPU availability:", flush=True)
    print(f"  CUDA available : {gpu_available}", flush=True)
    print(f"  Num GPUs       : {n_gpus}", flush=True)
    if gpu_available:
        print(f"  GPU name       : {gpu_name}", flush=True)
    else:
        print("  (Using CPU only)", flush=True)

    # -- Seed --------------------------------------------------------------
    L.seed_everything(42, workers=True)

    # -- DataModule --------------------------------------------------------
    # Resolve preload device
    preload_device = None
    if args.preload is not None:
        if args.preload == "auto":
            preload_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            preload_device = args.preload
        print(f"  Preloading dataset to: {preload_device}", flush=True)

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
        augment=args.augment,
        scale_range=tuple(args.scale_range),
        shift_range=tuple(args.shift_range),
        sabbarese_mode=args.sabbarese_mode,
    )
    dm.cache_dir = args.cache_dir if args.cache_dir else None
    dm.setup("fit")  # so we can read num_classes before model init

    # -- Model -------------------------------------------------------------
    arch_kw = _build_arch_kwargs(args)
    model = GestureClassifier(
        arch=args.arch,
        input_dim=dm.feature_dim,
        num_classes=dm.num_classes,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        **arch_kw,
    )

    # -- Print model summary -----------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Architecture : {args.arch}")
    print(f"  Parameters   : {n_params:,} ({n_trainable:,} trainable)")
    print(f"  Input dim    : {dm.feature_dim}")
    print(f"  Num classes  : {dm.num_classes}")
    print(f"  Seq length   : {args.seq_len}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Precision    : {args.precision}")
    print()

    # -- Per-run directory -------------------------------------------------
    # Find next available index for this architecture
    arch_base = args.arch
    base_dir = args.save_dir  # e.g., ./jester_runs
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if d.startswith(arch_base + "-") and os.path.isdir(os.path.join(base_dir, d))
    ]
    indices = []
    for d in existing:
        try:
            idx = int(d.split("-")[1])
            indices.append(idx)
        except (IndexError, ValueError):
            continue
    next_idx = max(indices, default=0) + 1
    run_name = args.run_name or f"{arch_base}-{next_idx}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run directory: {run_dir}")

    # -- Logger ------------------------------------------------------------
    if args.no_wandb:
        logger = None
    else:
        wandb_kwargs: dict = {
            "project": args.project,
            "name": run_name,
            "save_dir": run_dir,
            "log_model": "all",
        }
        if args.wandb_run_id:
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
        logger = WandbLogger(**wandb_kwargs)
        logger.experiment.config.update(vars(args))

    # -- Callbacks ---------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="best",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=args.early_stop_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # -- Trainer -----------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # GPU if available, else CPU
        devices="auto",
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip,
        deterministic=True,
        log_every_n_steps=10,
    )

    # -- Train -------------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    # -- Report best checkpoint --------------------------------------------
    ckpt_cb = trainer.checkpoint_callback
    best_ckpt = None
    if ckpt_cb and hasattr(ckpt_cb, "best_model_path"):
        best_ckpt = ckpt_cb.best_model_path
        print(f"\n  Best checkpoint: {best_ckpt}")
        print(f"  Best val/acc:    {ckpt_cb.best_model_score}\n")

    # -- Auto-evaluate ---------------------------------------------------
    if not args.skip_eval:
        eval_ckpt = best_ckpt or os.path.join(run_dir, "checkpoints", "last.ckpt")
        if os.path.exists(eval_ckpt):
            # Build eval output dir inside the run's own dir
            eval_out = os.path.join(run_dir, "eval")

            # Run info to associate results with this training run
            run_info = {
                "arch": args.arch,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "scheduler": args.scheduler,
                "label_smoothing": args.label_smoothing,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
                "max_hands": args.max_hands,
                "precision": args.precision,
                "wandb_project": args.project,
                "wandb_run_id": (
                    logger.experiment.id
                    if logger and hasattr(logger, "experiment")
                    else None
                ),
                "wandb_run_name": (
                    logger.experiment.name
                    if logger and hasattr(logger, "experiment")
                    else None
                ),
                "best_val_acc": (
                    float(ckpt_cb.best_model_score)
                    if ckpt_cb
                    and hasattr(ckpt_cb, "best_model_score")
                    and ckpt_cb.best_model_score is not None
                    else None
                ),
            }

            # Build a minimal argparse namespace for evaluate()
            import copy

            eval_args = copy.copy(args)
            eval_args.ckpt = eval_ckpt
            eval_args.output = eval_out
            eval_args.no_show = True  # non-interactive
            eval_args.split = "val"
            eval_args.normalize = "true"
            eval_args.top_n = 15

            print("\n" + "=" * 60)
            print("  Auto-evaluating after training...")
            print("=" * 60)
            evaluate(eval_args, run_info=run_info)
        else:
            print(f"  Skipping eval: checkpoint not found at {eval_ckpt}")


if __name__ == "__main__":
    train()
