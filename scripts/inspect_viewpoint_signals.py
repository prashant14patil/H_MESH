"""Print pose landmarks for selected frames without exporting images."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.capture_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((args.capture_dir / "processed_frames" / "processing_summary.json").read_text(encoding="utf-8"))
    times = {Path(f["filename"]).stem: f["timestamp_sec"] for f in metadata["frames"]}
    for frame in summary["frames"]:
        t = times.get(frame["frame"])
        if t is None or not (1.7 <= t <= 3.2 or 4.9 <= t <= 6.7):
            continue
        landmarks = frame.get("landmarks", {})
        names = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        values = {name: {k: landmarks.get(name, {}).get(k) for k in ("x", "z", "visibility", "visible")} for name in names}
        print(json.dumps({"frame": frame["frame"], "seconds": t, "landmarks": values}))


if __name__ == "__main__":
    main()
