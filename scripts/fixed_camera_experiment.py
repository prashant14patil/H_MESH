"""Experimental fixed-camera calibration and capture validation CLI."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.services.checkerboard_calibration import (board_points, calibrate_images,
    calibrate_observations, load_calibration, save_calibration)
from backend.services.fixed_camera_validation import validate_capture


def visualize(output, result, calibration=None, images=None, frame_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.set_title("Fixed camera assumption (not verified)")
    ax.text(.05, .6, "RGB camera\nstationary", bbox={"facecolor": "lightblue"})
    ax.annotate("", xy=(.8, .65), xytext=(.35, .65), arrowprops={"arrowstyle": "->"})
    ax.text(.72, .6, "person rotates\nin place (assumed)", bbox={"facecolor": "lightyellow"})
    ax.set_axis_off()
    frames = result["frames"]
    ax = axes[0, 1]
    ax.set_title("Selected frame sequence: heuristic orientation")
    ax.set_axis_off()
    lines = [f"{f['filename']}  t={f.get('timestamp_sec')}  {f['orientation']['label']}  score={f['orientation']['heuristic_score']}"
             for f in frames]
    ax.text(0, 1, "\n".join(lines[:20]) or "No selected frames", va="top", fontsize=8)
    ax = axes[1, 0]
    counts = result["coverage"]["heuristic_counts"]
    ax.bar(range(len(counts)), list(counts.values()))
    ax.set_xticks(range(len(counts)), list(counts), rotation=35, ha="right")
    ax.set_title("Coverage unconfirmed: heuristic counts")
    ax = axes[1, 1]
    ax.axis("off")
    warnings = [f"{f['filename']}: {', '.join(f['warnings'])}" for f in frames if f["warnings"]]
    ax.text(0, 1, "Quality warnings\n" + ("\n".join(warnings[:12]) or "None reported") +
            "\n\n" + result["coverage"]["warning"], va="top", wrap=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "capture_validation.png", dpi=140)
    plt.close(fig)
    if frames:
        count = len(frames)
        columns = min(4, count)
        rows = (count + columns - 1) // columns
        sheet, grid = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for i, axis in enumerate(grid.flat):
            axis.axis("off")
            if i >= count:
                continue
            frame = frames[i]
            filename = frame["filename"]
            if frame_dir is not None and Path(filename).name == filename:
                image = cv2.imread(str(frame_dir / filename))
                if image is not None:
                    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    axis.text(.5, .5, "Frame unreadable", ha="center", va="center")
            else:
                axis.text(.5, .5, "Image not supplied", ha="center", va="center")
            axis.set_title(f"{filename}\nt={frame.get('timestamp_sec')} | {frame['orientation']['label']}", fontsize=8)
        sheet.tight_layout()
        sheet.savefig(output / "frame_sequence.png", dpi=140)
        plt.close(sheet)
    fig, ax = plt.subplots(figsize=(9, 4))
    if calibration is None:
        ax.axis("off")
        ax.text(.1, .5, "Real camera calibration pending: no checkerboard images supplied")
    else:
        ax.plot(calibration["per_view_reprojection_px"], marker="o")
        ax.set_title(f"Calibration residuals: {calibration['provenance']}")
        ax.set_xlabel("checkerboard view")
        ax.set_ylabel("RMS reprojection (px)")
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=140)
    plt.close(fig)
    if images and calibration:
        for index, name in enumerate(images):
            image = cv2.imread(str(name))
            if image is None:
                continue
            b = calibration["board"]
            found, points = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                        (b["columns"], b["rows"]))
            if found:
                cv2.drawChessboardCorners(image, (b["columns"], b["rows"]), points, found)
                cv2.imwrite(str(output / f"checkerboard_{index:03d}.png"), image)


def synthetic_demo():
    obj = board_points(9, 6, .025)
    matrix = np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]])
    distortion = np.array([-.08, .02, .001, -.002, 0.])
    corners = []
    for i in range(10):
        rvec = np.array([-.22 + .05*i, .16*np.sin(i), .05*np.cos(i)])
        tvec = np.array([-.14 + .025*i, -.07 + .015*(i % 3), .9 + .04*(i % 4)])
        corners.append(cv2.projectPoints(obj, rvec, tvec, matrix, distortion)[0])
    calibration = calibrate_observations(corners, (640, 480), 9, 6, .025)
    names = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle")
    frames, poses = [], []
    for i in range(4):
        filename = f"synthetic_{i:03d}.jpg"
        frames.append({"filename": filename, "timestamp_sec": float(i), "is_blurry": i == 2,
                       "blur_score": 20 if i == 2 else 200, "diff_score": None if i == 0 else 10})
        points = {n: {"x": .5, "y": .5, "z": 0., "visibility": .95, "visible": True} for n in names}
        points["left_shoulder"]["x"] = .35 if i % 2 == 0 else .49
        points["right_shoulder"]["x"] = .65 if i % 2 == 0 else .51
        points["nose"]["z"] = -.15 if i == 0 else .15
        poses.append({"frame": Path(filename).stem, "person_detected": True,
                      "full_body_visible": True, "landmarks": points})
    metadata = {"frames": frames, "rejected_frames": []}
    result = validate_capture(metadata, {"frames": poses}, [f["filename"] for f in frames], allow_blurry=True)
    result["camera"]["calibration"] = "synthetic_demo_only; real calibration pending"
    return result, calibration


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    demo = sub.add_parser("synthetic-demo")
    demo.add_argument("--output", type=Path, required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--columns", type=int, required=True)
    cal.add_argument("--rows", type=int, required=True)
    cal.add_argument("--square-size", type=float, required=True)
    cal.add_argument("--images", nargs="+", type=Path, required=True)
    val = sub.add_parser("validate")
    val.add_argument("--output", type=Path, required=True)
    val.add_argument("--metadata", type=Path, required=True)
    val.add_argument("--pose-summary", type=Path, required=True)
    val.add_argument("--frame", action="append", required=True)
    val.add_argument("--allow-blurry", action="store_true")
    val.add_argument("--calibration", type=Path)
    val.add_argument("--frame-dir", type=Path, help="Optional directory containing explicitly selected frame images")
    args = parser.parse_args()

    calibration = None
    if args.mode == "synthetic-demo":
        result, calibration = synthetic_demo()
    elif args.mode == "calibrate":
        calibration = calibrate_images(args.images, args.columns, args.rows, args.square_size)
        result = {"camera": {"motion": "not_measured"}, "frames": [],
                  "coverage": {"heuristic_counts": {"unknown": 0},
                               "warning": "No capture processed", "status": "unconfirmed"}}
    else:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        poses = json.loads(args.pose_summary.read_text(encoding="utf-8"))
        result = validate_capture(metadata, poses, args.frame, allow_blurry=args.allow_blurry)
        if args.calibration:
            info = metadata.get("video_info", {})
            size = (info.get("width"), info.get("height"))
            calibration = load_calibration(args.calibration, size)
            result["camera"]["calibration"] = calibration["provenance"]
    args.output.mkdir(parents=True, exist_ok=False)
    if calibration:
        save_calibration(calibration, args.output / "calibration.json")
    (args.output / "capture_validation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    visualize(args.output, result, calibration, args.images if args.mode == "calibrate" else None,
              args.frame_dir if args.mode == "validate" else None)
    print(json.dumps({"mode": args.mode, "output": str(args.output),
                      "calibration": calibration["provenance"] if calibration else "pending",
                      "coverage": result["coverage"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
