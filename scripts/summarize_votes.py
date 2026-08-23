import csv
from collections import Counter

file_path = "output/landmark_votes.csv"

total_votes = 0
overall_votes = Counter()
category_votes = {}

with open(file_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        if not row:
            continue

        total_votes += 1

        # Old format: video, frame, vote (3 cols)
        if len(row) == 3:
            vid, frame, vote = row
            category = "unknown (old format)"
        # New format: video, frame, category, vote (4 cols)
        elif len(row) == 4:
            vid, frame, category, vote = row
        else:
            continue

        overall_votes[vote] += 1

        if category not in category_votes:
            category_votes[category] = Counter()
        category_votes[category][vote] += 1

print(f"Total Votes: {total_votes}\n")

print("--- Overall Results ---")
for v, c in overall_votes.most_common():
    print(f"  {v}: {c}")

print("\n--- Breakdown by Category ---")
for cat, counts in category_votes.items():
    print(f"\n{cat}:")
    for v, c in counts.most_common():
        print(f"  {v}: {c}")
