import json
import os
import subprocess
import time

SCALES = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
ARCHS = ["conv1d", "transformer", "stgcn", "gru", "lstm"]

# We parse accuracy directly from the eval_summary.json file


def main():
    results = {arch: {} for arch in ARCHS}

    print("=" * 60)
    print("Starting Massive FLOP Benchmark Sweep")
    print("=" * 60)

    # We will log everything to a master JSON file
    results_file = "jester_runs/flop_sweep_results.json"

    # Ensure jester_runs exists
    os.makedirs("jester_runs", exist_ok=True)

    # Reload existing results if they exist
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                saved = json.load(f)
                for a in ARCHS:
                    if a in saved:
                        results[a].update(saved[a])
        except Exception:
            pass

    for scale in SCALES:
        print(f"\n\n{'=' * 40}")
        print(f"RUNNING SCALE: {scale}x")
        print(f"{'=' * 40}")

        for arch in ARCHS:
            run_name = f"{arch}_scale_{scale}x"
            run_dir = os.path.join("jester_runs", run_name)
            print(f"\n---> Starting {run_name} ...")

            eval_file = os.path.join(run_dir, "eval", "eval_summary.json")

            # Resume logic: if eval_summary.json exists and we already have accuracy, skip training
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                        acc = data.get("splits", {}).get("val", {}).get("accuracy", 0.0)
                        if acc == 0.0:
                            acc = data.get("run", {}).get("best_val_acc", 0.0)

                        if acc > 0.0:
                            print(
                                f"  [>] Resuming {run_name}: Already finished with {acc * 100:.2f}% acc"
                            )
                            results[arch][str(scale)] = {
                                "best_acc": acc,
                                "duration_mins": data.get("run", {}).get(
                                    "duration_mins", 0.0
                                ),
                            }
                            # Dump intermediate results
                            with open(results_file, "w") as f:
                                json.dump(results, f, indent=4)
                            continue
                except Exception as e:
                    print(f"  [!] Failed to parse {eval_file}: {e}")

            cmd = [
                ".\\.venv\\Scripts\\python.exe",
                "-m",
                "jester.training.train",
                "--arch",
                arch,
                "--flop-scale",
                str(scale),
                "--run-name",
                run_name,
                "--epochs",
                "100",  # Use the fixed 100 epochs from earlier decisions
                "--preload",
                "cuda",
            ]

            start_time = time.time()
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()

                # Direct output to a debug log file on disk
                debug_log_path = os.path.join("jester_runs", f"{run_name}_debug.log")
                with open(debug_log_path, "w") as log_file:
                    process = subprocess.Popen(
                        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env
                    )
                    process.wait()

                if process.returncode != 0:
                    print(
                        f"  [!] Run failed with exit code {process.returncode}. Setting acc=0.0"
                    )

            except Exception as e:
                print(f"  [!] Failed to run {run_name}: {e}")

            duration = (time.time() - start_time) / 60.0

            # Extract final accuracy
            acc = 0.0
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                        acc = data.get("splits", {}).get("val", {}).get("accuracy", 0.0)
                        if acc == 0.0:
                            acc = data.get("run", {}).get("best_val_acc", 0.0)
                except Exception as e:
                    print(f"  [!] Failed to parse final eval {eval_file}: {e}")

            results[arch][str(scale)] = {"best_acc": acc, "duration_mins": duration}

            # Dump intermediate results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)

    print("\n\n" + "=" * 60)
    print("SWEEP COMPLETE!")
    print(f"Results saved to {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
