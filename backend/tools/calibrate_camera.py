"""Folder-based camera calibration CLI. No model assets are used."""
import argparse
import json
import sys
from pathlib import Path

from backend.services.checkerboard_calibration import CalibrationFailure
from backend.services.real_camera_calibration import run_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate one camera from a folder of checkerboard photographs.",
        epilog=("Columns and rows are the number of INNER corners, not squares. "
                "Square size is the physical side length in any consistent unit. "
                "Use images from the capture camera at its capture resolution."),
    )
    parser.add_argument("--input", required=True, type=Path, help="Folder of JPG, JPEG or PNG images")
    parser.add_argument("--columns", required=True, type=int, help="Number of inner corners across")
    parser.add_argument("--rows", required=True, type=int, help="Number of inner corners down")
    parser.add_argument("--square-size", required=True, type=float, help="Checkerboard square side length")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fixed_camera"))
    parser.add_argument("--run-id", help="Optional unique letters/digits/_/- identifier")
    parser.add_argument("--min-views", type=int, default=6)
    parser.add_argument("--warning-rms-px", type=float, default=0.5)
    parser.add_argument("--poor-rms-px", type=float, default=2.0)
    parser.add_argument("--high-image-error-px", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        output = run_calibration(
            args.input, args.columns, args.rows, args.square_size,
            output_root=args.output_root, run_id=args.run_id, min_views=args.min_views,
            warning_rms_px=args.warning_rms_px, poor_rms_px=args.poor_rms_px,
            high_image_error_px=args.high_image_error_px, sample_count=args.samples,
        )
    except (CalibrationFailure, OSError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    config = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    print(f"Run: {output}\nRMS: {config['rms_reprojection_px']:.4f} px\n"
          f"Screening: {config['quality_assessment']['status']}\n"
          "Independent real-camera validation remains necessary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
