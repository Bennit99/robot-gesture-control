#!/usr/bin/env python3
# Save RGB + Depth + metadata (intrinsics & odometry) for offline testing

import argparse
import csv
import os
import time
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_state import RobotStateClient
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Helpers
# -------------------------
def timestamp_ns(acq_time) -> int:
    # acq_time is a google.protobuf Timestamp
    return int(acq_time.seconds) * 1_000_000_000 + int(acq_time.nanos)


def se3_to_fields(T: SE3Pose):
    """Return (x,y,z,qw,qx,qy,qz,yaw) from a bosdyn SE3Pose."""
    x, y, z = T.x, T.y, T.z
    qw, qx, qy, qz = T.rot.w, T.rot.x, T.rot.y, T.rot.z
    yaw = T.rot.to_yaw()
    return x, y, z, qw, qx, qy, qz, yaw


def ensure_csv_with_header(csv_path: Path):
    header = [
        # timing & sources
        "timestamp_ns",
        "rgb_source",
        "depth_source",
        "width",
        "height",
        # intrinsics
        "l_fx",
        "l_fy",
        "l_cx",
        "l_cy",
        "r_fx",
        "r_fy",
        "r_cx",
        "r_cy",
        # odom_T_body
        "odom_x",
        "odom_y",
        "odom_z",
        "odom_qw",
        "odom_qx",
        "odom_qy",
        "odom_qz",
        "odom_yaw",
        # vision_T_body
        "vision_x",
        "vision_y",
        "vision_z",
        "vision_qw",
        "vision_qx",
        "vision_qy",
        "vision_qz",
        "vision_yaw",
        # body_T_cam_left (extrinsic)
        "body_cam_tx",
        "body_cam_ty",
        "body_cam_tz",
        "body_cam_qw",
        "body_cam_qx",
        "body_cam_qy",
        "body_cam_qz",
        # body_T_cam_right (extrinsic)
        "r_body_cam_tx",
        "r_body_cam_ty",
        "r_body_cam_tz",
        "r_body_cam_qw",
        "r_body_cam_qx",
        "r_body_cam_qy",
        "r_body_cam_qz",
        # depth units
        "depth_unit",
        "mm_per_unit",
        # file paths
        "rgb_path_left",
        "depth_path_left",
        "rgb_path_right",
        "depth_path_right",
    ]
    new_file = not csv_path.exists()
    if new_file:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(csv_path: Path, row):
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# -------------------------
# Main capture
# -------------------------
def main():
    print("start recording")
    period = 1.0 / max(1e-6, 10)
    frames = 0
    time_stamp = time.time()

    while True:
        tick = time.monotonic()
        ap = argparse.ArgumentParser(
            description="Capture RGB+Depth and save intrinsics/odometry to CSV."
        )
        ap.add_argument("--out", default="spot_captures", help="Output folder")
        ap.add_argument(
            "--rgb_source", default="frontleft_fisheye_image", help="Visual source"
        )
        ap.add_argument(
            "--depth_source",
            default="frontleft_depth_in_visual_frame",
            help="Depth source (aligned to visual)",
        )
        args = ap.parse_args()

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"captures_{time_stamp}.csv"
        ensure_csv_with_header(csv_path)

        # --- Connect to Spot
        ROBOT_IP = "192.168.10.3"  # os.getenv("SPOT_IP")
        ROBOT_USERNAME = os.getenv("SPOT_USERNAME")
        ROBOT_PASSWORD = os.getenv("SPOT_PASSWORD")
        print("ROBOT_USERNAME", ROBOT_USERNAME)

        sdk = bosdyn.client.create_standard_sdk("image_depth_plus_visual")
        robot = sdk.create_robot(ROBOT_IP)
        robot.authenticate(ROBOT_USERNAME, ROBOT_PASSWORD)
        bosdyn.client.util.authenticate(robot)

        image_client = robot.ensure_client(ImageClient.default_service_name)
        # (state_client is not strictly needed because we’ll use the image snapshot,
        # but we create it anyway if you want to query state elsewhere.)
        _ = robot.ensure_client(RobotStateClient.default_service_name)

        # --- Grab images (depth aligned to visual)
        sources = [
            "frontleft_depth_in_visual_frame",
            "frontleft_fisheye_image",
            "frontright_depth_in_visual_frame",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
            "back_fisheye_image",
        ]
        responses = image_client.get_image_from_sources(sources)
        if len(responses) < 2:
            raise RuntimeError("Failed to get both depth and visual images.")

        depth_ir_left = responses[0]
        rgb_ir_left = responses[1]
        depth_ir_right = responses[2]
        rgb_ir_right = responses[3]
        rgb_left = responses[4]
        rgb_right = responses[5]
        rgb_back = responses[6]

        # --- Decode images
        # Depth: uint16 PNG buffer -> numpy
        depth_u16_left = np.frombuffer(depth_ir_left.shot.image.data, dtype=np.uint16)
        depth_u16_left = depth_u16_left.reshape(
            depth_ir_left.shot.image.rows, depth_ir_left.shot.image.cols
        )

        depth_u16_right = np.frombuffer(depth_ir_right.shot.image.data, dtype=np.uint16)
        depth_u16_right = depth_u16_right.reshape(
            depth_ir_right.shot.image.rows, depth_ir_right.shot.image.cols
        )

        # Visual: JPEG buffer -> BGR
        bgr_front_left = cv2.imdecode(
            np.frombuffer(rgb_ir_left.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr_front_left is None:
            raise RuntimeError("Failed to decode visual image.")

        # Visual: JPEG buffer -> BGR
        bgr_front_right = cv2.imdecode(
            np.frombuffer(rgb_ir_right.shot.image.data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        # Visual: JPEG buffer -> BGR
        bgr_left = cv2.imdecode(
            np.frombuffer(rgb_left.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        # Visual: JPEG buffer -> BGR
        bgr_right = cv2.imdecode(
            np.frombuffer(rgb_right.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        # Visual: JPEG buffer -> BGR
        bgr_back = cv2.imdecode(
            np.frombuffer(rgb_back.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        H, W = bgr_front_left.shape[:2]

        # --- Intrinsics from visual (same camera model as aligned depth)
        model_left = rgb_ir_left.source.pinhole
        l_fx, l_fy = (
            model_left.intrinsics.focal_length.x,
            model_left.intrinsics.focal_length.y,
        )
        l_cx, l_cy = (
            model_left.intrinsics.principal_point.x,
            model_left.intrinsics.principal_point.y,
        )

        # --- Transforms @ image timestamp
        snap_left = rgb_ir_left.shot.transforms_snapshot
        print("snap", snap_left)
        cam_frame_left = rgb_ir_left.shot.frame_name_image_sensor

        model_right = rgb_ir_right.source.pinhole
        r_fx, r_fy = (
            model_right.intrinsics.focal_length.x,
            model_right.intrinsics.focal_length.y,
        )
        r_cx, r_cy = (
            model_right.intrinsics.principal_point.x,
            model_right.intrinsics.principal_point.y,
        )

        # --- Transforms @ image timestamp
        snap_right = rgb_ir_right.shot.transforms_snapshot
        print("snap", snap_right)
        cam_frame_right = rgb_ir_right.shot.frame_name_image_sensor

        odom_T_body = get_a_tform_b(snap_left, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        vision_T_body = get_a_tform_b(snap_left, VISION_FRAME_NAME, BODY_FRAME_NAME)
        body_T_cam_left = get_a_tform_b(snap_left, BODY_FRAME_NAME, cam_frame_left)
        body_T_cam_right = get_a_tform_b(snap_right, BODY_FRAME_NAME, cam_frame_right)

        # --- Timestamps
        ts_ns = timestamp_ns(rgb_ir_left.shot.acquisition_time)

        # --- Save images
        rgb_path_front_left = out_dir / f"rgb_front_left_{ts_ns}.jpg"
        rgb_path_front_right = out_dir / f"rgb_front_right_{ts_ns}.jpg"
        rgb_path_left = out_dir / f"rgb_left_{ts_ns}.jpg"
        rgb_path_right = out_dir / f"rgb_right_{ts_ns}.jpg"
        rgb_path_back = out_dir / f"rgb_back_{ts_ns}.jpg"
        depth_path_left = out_dir / f"depth_left_{ts_ns}.png"  # keep uint16
        depth_path_right = out_dir / f"depth_right_{ts_ns}.png"  # keep uint16
        cv2.imwrite(str(rgb_path_front_left), bgr_front_left)
        cv2.imwrite(str(rgb_path_front_right), bgr_front_right)
        cv2.imwrite(str(rgb_path_left), bgr_left)
        cv2.imwrite(str(rgb_path_right), bgr_right)
        cv2.imwrite(str(rgb_path_back), bgr_back)

        cv2.imwrite(str(depth_path_left), depth_u16_left)
        cv2.imwrite(str(depth_path_right), depth_u16_right)

        # --- Flatten transforms
        odom_vals = se3_to_fields(odom_T_body)  # x,y,z,qw,qx,qy,qz,yaw
        vision_vals = se3_to_fields(vision_T_body)  # x,y,z,qw,qx,qy,qz,yaw
        body_cam_vals_left = (
            body_T_cam_left.x,
            body_T_cam_left.y,
            body_T_cam_left.z,
            body_T_cam_left.rot.w,
            body_T_cam_left.rot.x,
            body_T_cam_left.rot.y,
            body_T_cam_left.rot.z,
        )
        body_cam_vals_right = (
            body_T_cam_right.x,
            body_T_cam_right.y,
            body_T_cam_right.z,
            body_T_cam_right.rot.w,
            body_T_cam_right.rot.x,
            body_T_cam_right.rot.y,
            body_T_cam_right.rot.z,
        )

        # Depth units in Spot depth PNGs are typically millimeters
        depth_unit = "mm"
        mm_per_unit = (
            1  # if you later see meters already, set to 1000 or adjust your reader
        )

        # --- CSV row
        row = [
            ts_ns,
            rgb_ir_left.source.name,
            depth_ir_left.source.name,
            W,
            H,
            l_fx,
            l_fy,
            l_cx,
            l_cy,
            r_fx,
            r_fy,
            r_cx,
            r_cy,
            *odom_vals,
            *vision_vals,
            *body_cam_vals_left,
            *body_cam_vals_right,
            depth_unit,
            mm_per_unit,
            str(rgb_path_left),
            str(depth_path_left),
            str(rgb_path_right),
            str(depth_path_right),
        ]
        append_row(csv_path, row)

        print("✅ Saved:")
        print("  RGB  ->", rgb_path_left)
        print("  Depth->", depth_path_left)
        print("  CSV  ->", csv_path)
        print(f"Intrinsics: fx={l_fx:.2f}, fy={l_fy:.2f}, cx={l_cx:.2f}, cy={l_cy:.2f}")
        print(
            f"ODOM pose @ image time: x={odom_vals[0]:.3f}, y={odom_vals[1]:.3f}, yaw={odom_vals[-1]:.3f} rad"
        )
        print(
            f"VISION pose @ image time: x={vision_vals[0]:.3f}, y={vision_vals[1]:.3f}, yaw={vision_vals[-1]:.3f} rad"
        )

        frames += 1
        # pacing
        sleep_left = period - (time.monotonic() - tick)
        if sleep_left > 0:
            time.sleep(sleep_left)

    return True


if __name__ == "__main__":
    print("main")
    import sys

    if not main():
        sys.exit(1)
