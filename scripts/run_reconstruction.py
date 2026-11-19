"""Run the unified 360-degree reconstruction on an uploaded recording (no web server needed).

    .venv\Scripts\python.exe scripts/run_reconstruction.py <video_id> [--person N]

Uses the real HMR2 model. Prints progress and the final report location.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from backend.services import body_reconstruction as br  # noqa: E402
from backend.services.inference_engine import get_engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--person", type=int, default=None, help="candidate id if several people are detected")
    ap.add_argument("--sex", choices=["male", "female"], default=None, help="sex-specific body-shape prior")
    ap.add_argument("--height-cm", type=float, default=None, help="barefoot height for cm measurements")
    args = ap.parse_args()
    engine = get_engine()
    engine._load()
    if not engine.is_ready:
        sys.exit(f"HMR2 not ready: {engine.load_error}")
    run = br.start_run(args.video_id, height_cm=args.height_cm, sex=args.sex)
    run_id, last = run["run_id"], None
    while True:
        r = br.get_run(run_id)
        line = f"{r['status']:>26} {r['stage']:>8} {r['progress']*100:5.1f}%  {r['message']}"
        if line != last:
            print(line, flush=True)
            last = line
        if r["status"] == "awaiting_person_selection":
            if args.person is None:
                print("Candidates:", json.dumps(r["candidates"], indent=1))
                sys.exit("Re-run with --person <id>")
            br.select_person(run_id, args.person)
        if r["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    print(json.dumps(r.get("result") or r.get("error"), indent=2))
    print("outputs:", br._run_dir(run_id).resolve())


if __name__ == "__main__":
    main()
