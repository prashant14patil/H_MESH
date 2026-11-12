"""Diagnose pose/viewpoint output on sequentially decoded video frames.

This bypasses the production frame extractor and is NOT an end-to-end test.
The output directory contains sensitive copies of body images and results.
Example:
  python scripts/validate_pose_views.py data/captures/ID/original.webm data/pose_validation/views
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.output.resolve()
    video_id = "sequential_views"
    frames_dir = root / video_id / "extracted_frames"
    if frames_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing directory: {frames_dir}")
    frames_dir.mkdir(parents=True)
    os.environ["CAPTURE_OUTPUT_DIR"] = str(root)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    timestamps = {}
    next_ms = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ts >= next_ms:
                name = f"frame_{len(timestamps) + 1:03d}"
                if not cv2.imwrite(str(frames_dir / f"{name}.jpg"), frame):
                    raise RuntimeError(f"Failed to write {name}")
                timestamps[name] = round(ts / 1000, 3)
                next_ms += 1000
    finally:
        cap.release()

    if not timestamps:
        raise RuntimeError("No frames decoded")
    from backend.services.pose_processor import get_pose_engine
    from backend.services.viewpoint_estimator import estimate_viewpoints

    engine = get_pose_engine()
    print("Pose job:", engine._process_frames(video_id), flush=True)
    summary = engine.get_summary(video_id)
    viewpoints = estimate_viewpoints(video_id)
    by_frame = {item["frame"]: item for item in viewpoints["frames"]}
    for frame in summary["frames"]:
        view = by_frame[frame["frame"]]
        print(json.dumps({
            "frame": frame["frame"],
            "timestamp_sec": timestamps[frame["frame"]],
            "person_detected": frame.get("person_detected"),
            "landmark_count": len(frame.get("landmarks", {})),
            "full_body_visible": frame.get("full_body_visible"),
            "status": frame.get("status"),
            "viewpoint": view.get("viewpoint"),
            "confidence": view.get("confidence"),
            "reason": view.get("reason"),
        }), flush=True)
    print("Coverage:", json.dumps(viewpoints["coverage"]), flush=True)
    print("Labels require comparison with the actual decoded frames.")


if __name__ == "__main__":
    main()
