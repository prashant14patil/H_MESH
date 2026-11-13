"""Export authorized HMR2 diagnostics for one explicitly boxed frame."""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from backend.services.hmr2_diagnostics import export_frame
from backend.services.inference_engine import get_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    engine = get_engine()
    deadline = time.monotonic() + args.timeout
    while not engine.is_ready and engine.load_error is None and time.monotonic() < deadline:
        time.sleep(.25)
    if not engine.is_ready:
        raise RuntimeError(engine.load_error or "Model load timed out")
    result = export_frame(engine, args.frame, args.bbox, args.output, args.frame_id)
    print("Diagnostic saved:", args.output, "frame:", result["frame_id"])


if __name__ == "__main__":
    main()
