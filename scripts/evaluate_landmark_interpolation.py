import json
import subprocess
from pathlib import Path

import pandas as pd

# Paths
LABELS_DIR = Path("data/jester/labels")
EVAL_OUTPUT_DIR = Path("output/ablation_study")

EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run ablation
results = []
gaps_to_test = [0, 1, 2, 3, 4, 5, 6, 7]

print("\nRunning ablation study on full dataset...")
for gap in gaps_to_test:
    print(f"\n--- Testing Max Gap: {gap} ---")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "jester.training.evaluate",
        "--ckpt",
        f"models/gru-12_ml9_gap{gap}.ckpt",
        "--labels",
        str(LABELS_DIR),
        "--output",
        str(EVAL_OUTPUT_DIR),
        "--no-show",
        "--min-landmarks",
        "9",
    ]

    if gap == 0:
        cmd.append("--no-interpolate-missing")
    else:
        cmd.extend(["--interpolate-missing", "--interpolate-max-gap", str(gap)])

    # Run the evaluate script and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse accuracy from json
    acc = 0.0
    json_path = EVAL_OUTPUT_DIR / "eval_summary.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            acc = data.get("accuracy", 0.0) * 100

    if acc == 0.0:
        print("Failed to run evaluation or parse accuracy!")
        if result.stderr:
            print("Error:", result.stderr)

    print(f"Max Gap {gap} -> Accuracy: {acc}%")
    results.append({"Gap": gap, "Accuracy": acc})

# Print summary table
print("\n=== ABLATION RESULTS ===")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# Save results to JSON cache
out_json = EVAL_OUTPUT_DIR / "interpolation_ablation_results.json"
with open(out_json, "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nSaved results to {out_json}")

# Optional: find when it starts decreasing
prev_acc = -1
for r in results:
    if r["Accuracy"] < prev_acc:
        print(f"\nAccuracy started decreasing at gap {r['Gap']}! Best was {prev_acc}%")
        break
    prev_acc = r["Accuracy"]
