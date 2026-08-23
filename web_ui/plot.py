import math
import os

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2

from cfg import cfg
from utils.mediapipe_extensions import MyGestureRecognizerResult

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.labelbottom": False,
        "xtick.bottom": False,
        "ytick.labelleft": False,
        "ytick.left": False,
        "xtick.labeltop": False,
        "xtick.top": False,
        "ytick.labelright": False,
        "ytick.right": False,
    }
)

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore


def plot(images, results: list[MyGestureRecognizerResult]):
    """Plots images or videos with the gesture category and its score along with the hand landmarks."""
    if cfg.running_mode == "IMAGE":
        plot_images(images, results)
    elif cfg.running_mode == "VIDEO":
        plot_videos(images, results)


def plot_images(images, results: list[MyGestureRecognizerResult]):
    """plots images with the gesture category and its score along with the hand landmarks in batches."""
    for i in range(0, len(images), cfg.batch_size):
        plot_batch_of_images(
            images[i : i + cfg.batch_size], results[i : i + cfg.batch_size]
        )


def plot_batch_of_images(images, results: list[MyGestureRecognizerResult]):
    """plots a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # plot gestures and hand landmarks.
    for i in range(rows * cols):
        annotated_image = annotate_image(images[i], results[i])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(annotated_image)

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def plot_videos(images, results):
    # Images and labels.
    images = [image.numpy_view() for image in images]
    # Annotate images
    annotated_images = [
        annotate_image(images[i], results[i]) for i in range(len(images))
    ]

    # Get video properties
    height, width = annotated_images[0].shape[:2]

    video_path = os.path.join(cfg.output, "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(video_path, fourcc, cfg.video_fps, (width, height))

    # Write frames to video
    for frame in annotated_images:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    # Play the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Annotated Gesture Video", frame)
        if cv2.waitKey(int(1000 / cfg.video_fps)) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def annotate_image(
    image, recognition_result: MyGestureRecognizerResult, landmarks=True, labels=True
):
    """Draws hand landmarks and gesture labels on the image and returns the annotated image."""
    annotated_image = image.copy()

    # Draw landmarks
    if landmarks:
        multi_hand_landmarks_list = recognition_result.hand_landmarks
        for hand_landmarks in multi_hand_landmarks_list:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(  # type: ignore
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    # Draw gesture labels in the top left corner, stacked vertically
    if labels:
        gestures_str = recognition_result.gestures_str()
        _, w = annotated_image.shape[:2]
        y_offset = 10  # Start a bit below the top edge
        for idx, text in enumerate(gestures_str):
            # Dynamically adjust font size so label width is between 40% and 70% of image width
            max_w = 0.6  # Maximum label width as a fraction of image width
            min_w = 0.25  # Minimum label width as a fraction of image width
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # compute scaling factor
            target_width = min(max(w * max_w, text_width), w * max_w)
            if text_width > 0:
                scale_factor = target_width / text_width
                font_scale = max(0.2, min(scale_factor, 3.0))
            else:
                font_scale = 1.0
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # Ensure label is at least min_w of image width
            if text_width < int(w * min_w):
                font_scale = max(font_scale, (w * min_w) / max(text_width, 1))
                font_scale = min(font_scale, 3.0)
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
            # Place first label at y_offset + text_height, stack subsequent labels below
            y = y_offset + int(text_height * 1.2) + idx * int(text_height * 1.5)
            cv2.putText(
                annotated_image,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
    return annotated_image
