"""Inspect sequential decoding and write a diagnostic contact sheet.

Usage: python scripts/inspect_video_timestamps.py INPUT OUTPUT.jpg
The output contains potentially sensitive frames from the supplied video.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    thumbnails = []
    target_ms = 0.0
    count = 0
    last_ms = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            last_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if last_ms >= target_ms:
                image = cv2.resize(frame, (320, 180))
                cv2.putText(image, f"{last_ms / 1000:.2f}s", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                thumbnails.append(image)
                target_ms += 1000.0
    finally:
        cap.release()
    if not thumbnails:
        raise RuntimeError("No frames decoded")
    blank = np.zeros_like(thumbnails[0])
    rows = [np.hstack((thumbnails + [blank, blank])[i:i + 3])
            for i in range(0, len(thumbnails), 3)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(rows)):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"Sequential frames: {count}; last timestamp: {last_ms / 1000:.3f}s")
    print(f"Contact sheet: {args.output} ({len(thumbnails)} samples)")


if __name__ == "__main__":
    main()
