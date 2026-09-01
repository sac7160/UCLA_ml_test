"""
extract_fingertip_trajectory.py

Re-track the index fingertip directly from the raw video (camera2_raw.avi)
using MediaPipe Hands, with Savitzky-Golay smoothing to remove the per-frame
jitter present in the trajectory.csv recorded live during collection.

Usage:
    # Single trial
    python extract_fingertip_trajectory.py \
        --video dataset/p1/dataset/d/trial_001/camera2_raw.avi \
        --out dataset/p1/dataset/d/trial_001/trajectory_smooth.csv

    # Every trial for one label
    python extract_fingertip_trajectory.py --dataset-root dataset --label d --batch

    # Every trial for EVERY label (whole dataset) -- just omit --label
    python extract_fingertip_trajectory.py --dataset-root dataset --batch

    # Whole dataset, but only specific participants
    python extract_fingertip_trajectory.py --dataset-root dataset --batch --participants p1 p2
"""
import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

INDEX_FINGERTIP_LANDMARK = 8  # MediaPipe Hands landmark index for the index fingertip


def track_video(video_path, min_detection_conf=0.5, min_tracking_conf=0.5):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    times, xs, ys, detected = [], [], [], []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=min_detection_conf,
                         min_tracking_confidence=min_tracking_conf) as hands:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            h, w = frame.shape[:2]
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark[INDEX_FINGERTIP_LANDMARK]
                xs.append(lm.x * w)
                ys.append(lm.y * h)
                detected.append(1)
            else:
                xs.append(np.nan)
                ys.append(np.nan)
                detected.append(0)
            times.append(frame_idx / fps)
            frame_idx += 1
    cap.release()
    return np.array(times), np.array(xs), np.array(ys), np.array(detected)


def smooth(x, y, detected, window=9, polyorder=2):
    x, y = x.copy(), y.copy()
    valid = detected.astype(bool)
    if valid.sum() < 3:
        return x, y

    idx = np.arange(len(x))
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])

    w = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if w >= 5:
        x = savgol_filter(x, w, polyorder)
        y = savgol_filter(y, w, polyorder)
    return x, y


def process_one(video_path, out_path):
    t, x, y, detected = track_video(video_path)
    x_s, y_s = smooth(x, y, detected)
    pd.DataFrame({
        "time_aligned": t, "x_px": x_s, "y_px": y_s, "detected": detected,
    }).to_csv(out_path, index=False)
    print(f"[OK] {video_path} -> {out_path}  ({int(detected.sum())}/{len(detected)} frames detected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to a single camera2_raw.avi")
    ap.add_argument("--out", help="Output csv path for a single video")
    ap.add_argument("--dataset-root")
    ap.add_argument("--label", default=None,
                     help="Restrict batch processing to one label. Omit to process ALL labels.")
    ap.add_argument("--participants", nargs="*", default=None,
                     help="Restrict batch processing to specific participant IDs")
    ap.add_argument("--batch", action="store_true",
                     help="Process every matching trial under dataset-root")
    ap.add_argument("--filename", default="trajectory_smooth_120hz.csv",
                     help="Output filename when batching (saved inside each trial folder)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="Skip trials that already have the output file")
    args = ap.parse_args()

    if args.batch:
        if not args.dataset_root:
            raise SystemExit("--batch requires --dataset-root")
        label_pattern = args.label if args.label else "*"
        videos = sorted(glob.glob(os.path.join(
            args.dataset_root, "*", "dataset", label_pattern, "trial_*", "camera_raw.avi"))) #"camera2_raw.avi")))

        if args.participants:
            videos = [v for v in videos if any(f"{os.sep}{p}{os.sep}" in v for p in args.participants)]

        desc = f"label='{args.label}'" if args.label else "ALL labels"
        print(f"[INFO] Found {len(videos)} videos for {desc}"
              f"{' (participants=' + ','.join(args.participants) + ')' if args.participants else ''}")

        n_done, n_skipped, n_failed = 0, 0, 0
        for v in videos:
            out_path = os.path.join(os.path.dirname(v), args.filename)
            if args.skip_existing and os.path.exists(out_path):
                n_skipped += 1
                continue
            try:
                process_one(v, out_path)
                n_done += 1
            except Exception as e:
                print(f"[FAIL] {v}: {e}")
                n_failed += 1
        print(f"[SUMMARY] processed={n_done} skipped={n_skipped} failed={n_failed}")
    else:
        if not (args.video and args.out):
            raise SystemExit("Provide --video and --out, or use --batch")
        process_one(args.video, args.out)


if __name__ == "__main__":
    main()
