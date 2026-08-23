import random
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

ROOT = Path(r"c:\Users\stu245938\Desktop\robot-gesture-control")
L_DIR = ROOT / "data" / "jester" / "landmarks"
Y_DIR = ROOT / "data" / "jester" / "landmarks_yolo"
IMG_DIR = ROOT / "data" / "jester" / "images"
VOTES_FILE = ROOT / "output" / "landmark_votes.csv"

DIST_THRESHOLD = 0.3
SCALE_FACTOR = 2  # Make window larger


def draw_landmarks_mp(img, df, frame_idx):
    frame_df = df[df["frame"] == frame_idx]

    # MediaPipe drawing requires a proto
    # Group by hand_index
    hands = frame_df["hand_index"].unique()
    for h in hands:
        hand_df = frame_df[frame_df["hand_index"] == h]
        if hand_df.empty:
            continue

        row = hand_df.iloc[0]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks = []
        for i in range(21):
            x_col = f"x{i}"
            y_col = f"y{i}"
            z_col = f"z{i}"
            if x_col in row and y_col in row:
                landmarks.append(
                    landmark_pb2.NormalizedLandmark(
                        x=row[x_col],
                        y=row[y_col],
                        z=row[z_col] if z_col in row else 0.0,
                    )
                )
        if len(landmarks) == 21:
            hand_landmarks_proto.landmark.extend(landmarks)
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    return img


def main():
    print("Loading data...")
    mp_vids = [p.name for p in L_DIR.iterdir() if p.is_dir()]
    y_vids = [p.name for p in Y_DIR.iterdir() if p.is_dir()]
    common = list(set(mp_vids).intersection(set(y_vids)))
    random.shuffle(common)

    existing_votes = set()
    if VOTES_FILE.exists():
        votes_df = pd.read_csv(VOTES_FILE)
        # Handle old format (video,frame,vote) vs new format (video,frame,category,vote)
        if "category" not in votes_df.columns:
            votes_df["category"] = "unknown"
        for _, row in votes_df.iterrows():
            existing_votes.add(f"{row['video']}_{row['frame']}")
    else:
        VOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VOTES_FILE, "w") as f:
            f.write("video,frame,category,vote\n")

    print("\n=== Landmark Voting Tool ===")
    print("Controls:")
    print("  'm' : MediaPipe is better")
    print("  'y' : YOLO is better")
    print("  'b' : Both are bad")
    print("  'g' : Both are good")
    print("  's' : Skip")
    print("  'q' : Quit")
    print(f"Results saved to: {VOTES_FILE}")
    print("============================\n")

    # We want to randomly pick frames.
    # Since checking all 10000 videos takes too long, we will randomly sample a video,
    # find discrepant frames, and present one randomly.

    while True:
        vid = random.choice(common)
        try:
            mp_df = pd.read_parquet(L_DIR / vid / "landmarks.parquet")
            y_df = pd.read_parquet(Y_DIR / vid / "landmarks.parquet")
        except Exception:
            continue

        frames = set(mp_df["frame"]).union(set(y_df["frame"]))
        frames = sorted(list(frames))

        # Find all discrepancies in this video
        discrepant_frames = []
        for frame in frames:
            if f"{vid}_{frame}" in existing_votes:
                continue

            mp_frame_df = mp_df[mp_df["frame"] == frame]
            y_frame_df = y_df[y_df["frame"] == frame]

            category = None
            if len(mp_frame_df) > len(y_frame_df):
                category = "mp_more_hands"
            elif len(y_frame_df) > len(mp_frame_df):
                category = "yolo_more_hands"
            else:
                merged = pd.merge(
                    mp_frame_df,
                    y_frame_df,
                    on=["frame", "hand_index"],
                    suffixes=("_mp", "_yolo"),
                )
                if not merged.empty:
                    dists = []
                    for lm in range(21):
                        dx = merged[f"x{lm}_mp"] - merged[f"x{lm}_yolo"]
                        dy = merged[f"y{lm}_mp"] - merged[f"y{lm}_yolo"]
                        dists.append(np.sqrt(dx**2 + dy**2))
                    mean_dists = np.mean(np.stack(dists), axis=0)
                    if np.any(mean_dists > DIST_THRESHOLD):
                        category = "different_locations"

            if category:
                discrepant_frames.append((frame, category, mp_frame_df, y_frame_df))

        if not discrepant_frames:
            continue

        # Pick one random discrepant frame from this video to ensure variety
        frame, category, mp_frame_df, y_frame_df = random.choice(discrepant_frames)

        # Load Image
        img_path = IMG_DIR / vid / f"{frame:05d}.jpg"
        if not img_path.exists():
            img_path = IMG_DIR / vid / f"{frame}.jpg"
            if not img_path.exists():
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_mp = img.copy()
        img_yolo = img.copy()

        img_mp = draw_landmarks_mp(img_mp, mp_frame_df, frame)
        img_yolo = draw_landmarks_mp(img_yolo, y_frame_df, frame)

        # Resize for better visibility
        if SCALE_FACTOR != 1:
            h, w = img_mp.shape[:2]
            img_mp = cv2.resize(img_mp, (w * SCALE_FACTOR, h * SCALE_FACTOR))
            img_yolo = cv2.resize(img_yolo, (w * SCALE_FACTOR, h * SCALE_FACTOR))

        # Add text
        cv2.putText(
            img_mp, "MediaPipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            img_yolo, "YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )

        cv2.putText(
            img_mp,
            f"Cat: {category}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        combined = np.hstack((img_mp, img_yolo))

        cv2.namedWindow("Landmark Comparison", cv2.WINDOW_NORMAL)
        cv2.imshow("Landmark Comparison", combined)

        key = cv2.waitKey(0) & 0xFF
        vote = None

        if key == ord("m"):
            vote = "mediapipe"
        elif key == ord("y"):
            vote = "yolo"
        elif key == ord("b"):
            vote = "both_bad"
        elif key == ord("g"):
            vote = "both_good"
        elif key == ord("q"):
            print("Quitting...")
            cv2.destroyAllWindows()
            return
        elif key == ord("s"):
            continue

        if vote:
            with open(VOTES_FILE, "a") as f:
                f.write(f"{vid},{frame},{category},{vote}\n")
            existing_votes.add(f"{vid}_{frame}")


if __name__ == "__main__":
    main()
