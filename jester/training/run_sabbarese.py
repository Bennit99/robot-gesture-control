import subprocess
import sys


def main():
    print("Starting Sabbarese Replication Run...")
    # Sabbarese used 16 frame windows, 64 hidden units, 2 layers, no dropping videos (min_landmarks=0)
    # We will use our newly added --sabbarese-mode flag.
    cmd = [
        sys.executable,
        "-m",
        "jester.training.train",
        "--arch",
        "gru",
        "--hidden_dim",
        "64",
        "--num_layers",
        "2",
        "--seq-len",
        "16",
        "--min-landmarks",
        "0",
        "--sabbarese-mode",
        "--epochs",
        "100",
        "--run-name",
        "gru_sabbarese_replication",
    ]

    # Execute the training
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
