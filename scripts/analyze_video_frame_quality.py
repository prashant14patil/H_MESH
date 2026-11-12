"""Read-only analysis of existing extraction and source video; no inference."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def lap(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1", type=Path)
    parser.add_argument("--diagnostic-threshold", type=float, default=100.0,
                        help="Existing native-resolution whole-frame threshold; for source comparison only")
    args = parser.parse_args()
    meta = json.loads((args.capture / "metadata.json").read_text(encoding="utf-8"))
    pose = json.loads((args.capture / "processed_frames" / "processing_summary.json").read_text(encoding="utf-8"))
    poses = {p["frame"]: p for p in pose["frames"]}
    v1 = set()
    if args.v1:
        v1 = {x["filename"] for x in json.loads(args.v1.read_text(encoding="utf-8"))["frames"]}
    rows = []
    for f in meta["frames"]:
        name = f["filename"]
        p = poses.get(Path(name).stem, {})
        image = cv2.imread(str(args.capture / "extracted_frames" / name))
        landmarks = p.get("landmarks") or {}
        points = [(float(l["x"]), float(l["y"])) for l in landmarks.values()
                  if l.get("visible") and isinstance(l.get("x"), (float, int))
                  and isinstance(l.get("y"), (float, int))
                  and 0 <= l["x"] <= 1 and 0 <= l["y"] <= 1]
        jpeg_score = resized_score = roi_score = None
        dimensions = None
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            dimensions = [width, height]
            jpeg_score = lap(gray)
            resized_score = lap(cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA))
            if points:
                x0 = max(0, int((min(x for x, _ in points) - .05) * width))
                y0 = max(0, int((min(y for _, y in points) - .05) * height))
                x1 = min(width, int((max(x for x, _ in points) + .05) * width))
                y1 = min(height, int((max(y for _, y in points) + .05) * height))
                if x1 > x0 and y1 > y0:
                    roi_score = lap(gray[y0:y1, x0:x1])
        reasons = []
        if image is None:
            reasons.append("unreadable")
        if f.get("is_blurry") is not False:
            reasons.append("blurry_or_unknown")
        if p.get("person_detected") is not True:
            reasons.append("person_not_detected_or_unknown")
        if p.get("full_body_visible") is not True:
            reasons.append("full_body_not_confirmed")
        rows.append({
            "filename": name, "timestamp_sec": f.get("timestamp_sec"),
            "resolution": dimensions, "extractor_laplacian_variance": f.get("blur_score"),
            "extractor_is_blurry": f.get("is_blurry"), "jpeg_laplacian_variance": jpeg_score,
            "jpeg_640x360_laplacian_variance": resized_score,
            "landmark_roi_laplacian_variance": roi_score,
            "person_detected": p.get("person_detected"),
            "person_detection_confidence": None,
            "person_detection_confidence_note": "Unavailable; pose detection_confidence is mean landmark visibility, not calibrated detection probability.",
            "mean_landmark_visibility": p.get("detection_confidence"),
            "full_body_visible": p.get("full_body_visible"),
            "landmarks": landmarks, "visible_in_image_landmark_count": len(points),
            "missing_regions": p.get("missing_regions"),
            "duplicate_status": "accepted_not_rejected_as_duplicate" if f.get("diff_score") is not None else "unknown_no_comparison",
            "extractor_diff_score": f.get("diff_score"),
            "selection_suitable": not reasons, "selection_reasons": reasons,
            "in_v1": name in v1,
        })
    sharpness_sorted = sorted(rows, key=lambda r: r["extractor_laplacian_variance"] if r["extractor_laplacian_variance"] is not None else float("-inf"), reverse=True)
    for i, row in enumerate(sharpness_sorted):
        row["sharpness_rank"] = i + 1
    rows.sort(key=lambda r: (bool(r["selection_suitable"]), r["full_body_visible"] is True,
                             r["mean_landmark_visibility"] or 0,
                             r["extractor_laplacian_variance"] or 0), reverse=True)
    for i, row in enumerate(rows):
        row["review_rank"] = i + 1
    videos = list(args.capture.glob("original.*"))
    if len(videos) != 1:
        raise ValueError("Expected exactly one original video")
    cap = cv2.VideoCapture(str(videos[0]))
    if not cap.isOpened():
        raise ValueError("Cannot decode source video")
    header = {"reported_fps": cap.get(cv2.CAP_PROP_FPS), "reported_frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
              "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    native, resized, passing = [], [], []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score = lap(gray)
            native.append(score)
            resized.append(lap(cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)))
            if score >= args.diagnostic_threshold:
                passing.append({"decoded_index_zero_based": len(native)-1,
                                "timestamp_sec": cap.get(cv2.CAP_PROP_POS_MSEC)/1000, "score": score})
    finally:
        cap.release()
    if not native:
        raise ValueError("No decodable source frames")
    v1_scores = [r["extractor_laplacian_variance"] for r in rows if r["in_v1"]]
    summary = {
        "capture": str(args.capture), "video_header": header, "stored_video_info": meta.get("video_info"),
        "decoded_count": len(native),
        "decoded_native_laplacian_min_median_max": [min(native), float(np.median(native)), max(native)],
        "decoded_resized_640x360_laplacian_min_median_max": [min(resized), float(np.median(resized)), max(resized)],
        "diagnostic_threshold_native": args.diagnostic_threshold,
        "decoded_frames_meeting_diagnostic_threshold": passing,
        "extracted_frames": len(rows), "extracted_non_blurry": sum(r["extractor_is_blurry"] is False for r in rows),
        "eligible_frames": sum(r["selection_suitable"] for r in rows),
        "v1_mean_extractor_laplacian": float(np.mean(v1_scores)) if v1_scores else None,
        "extractor_rejected_candidates": meta.get("rejected_frames", []),
        "ranking": "suitability, full-body visibility, mean landmark visibility, then stored native sharpness",
        "warnings": ["Whole-frame Laplacian variance depends on resolution, texture, compression and background; it does not isolate motion blur.",
                     "ROI score is a landmark-derived region and is diagnostic only; no ROI threshold is validated.",
                     "Person count and calibrated person detection confidence are unavailable.",
                     "Video frames and image-space data contain personal information."],
        "frames": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frame_quality.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output / "frame_quality.csv").open("x", encoding="utf-8", newline="") as fh:
        columns = ["review_rank", "sharpness_rank", "filename", "timestamp_sec", "resolution", "extractor_laplacian_variance",
                   "extractor_is_blurry", "jpeg_laplacian_variance", "jpeg_640x360_laplacian_variance",
                   "landmark_roi_laplacian_variance", "person_detected", "person_detection_confidence",
                   "mean_landmark_visibility", "full_body_visible", "visible_in_image_landmark_count",
                   "landmarks", "missing_regions", "duplicate_status", "extractor_diff_score",
                   "selection_suitable", "selection_reasons", "in_v1"]
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))
    print("ranking:", [(r["filename"], r["extractor_laplacian_variance"], r["full_body_visible"],
                        r["mean_landmark_visibility"]) for r in rows])


if __name__ == "__main__":
    main()
