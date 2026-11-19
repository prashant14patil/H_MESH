"""One command: 360-degree video -> unified 3D body mesh, measurements and quality reports.

    .venv\\Scripts\\python.exe scripts/scan_360.py VIDEO.mp4 --consent-attested [--height-cm 179]
    .venv\\Scripts\\python.exe scripts/scan_360.py --video-id <uploaded recording id> [--height-cm 179]

Stages (all in-process, no web server needed):
  ingest   validate + store the video, extract preview frames, record the operator's consent attestation
  decode -> detect -> track -> HMR2 -> rotation/viewpoints -> frame selection -> capture-quality report
         -> silhouettes -> multi-view shared-shape SMPL fit -> mesh validation -> measurements
  scale    centimetres ONLY if --height-cm (barefoot stature) is given; otherwise relative units
  shapy    optional SHAPY comparison on the same selected frames (--shapy)
  report   scan_report.html with the mesh renders, measurements, capture and reconstruction quality

Exit code 0 = mesh produced, 2 = reconstruction blocked (the capture feedback is printed), 1 = usage/setup error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # output roots are relative to the repository
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


def _ingest(video: Path, consent: bool) -> str:
    from backend.services import capture_authorization, video_processor
    from backend.utils.video_validators import VideoValidationError, validate_video
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    if not consent:
        sys.exit("Refusing to process: pass --consent-attested to confirm you are authorised to process this "
                 "recording and the person in it consented (same attestation as the upload form).")
    data = video.read_bytes()
    try:
        validate_video(filename=video.name, data=data)
    except VideoValidationError as exc:
        sys.exit(f"Video rejected: {exc}")
    video_id = str(uuid.uuid4())
    engine = video_processor.get_capture_engine()
    engine.save_video(video_id=video_id, filename=video.name, data=data)
    capture_authorization.attest(video_id, True, True)
    print(f"[ingest] recording {video_id}: extracting frames", flush=True)
    result = engine._extract_frames(video_id, str(engine._video_dir(video_id) / ("original" + (video.suffix.lower() or ".mp4"))))
    if result.get("status") != "completed":
        sys.exit(f"Frame extraction failed: {result.get('detail') or result}")
    return video_id


def _wait(br, run_id: str, person: int | None) -> dict:
    last = None
    while True:
        r = br.get_run(run_id)
        line = f"[{r['stage']:>7}] {r['progress'] * 100:5.1f}%  {r['message']}"
        if line != last:
            print(line, flush=True)
            last = line
        if r["status"] == "awaiting_person_selection":
            cands = r["candidates"]
            print("Several people detected:")
            for c in cands:
                print(f"  {c['id']}: {c['label']}  ({br._run_dir(run_id) / c['thumbnail']})")
            choice = person
            if choice is None and sys.stdin.isatty():
                choice = int(input("Candidate id to reconstruct: ").strip())
            if choice is None:
                sys.exit("Re-run with --person <id>")
            br.select_person(run_id, choice)
            person = choice
        if r["status"] in ("completed", "failed", "interrupted"):
            return r
        time.sleep(1)


def _print_summary(run_dir: Path, run: dict) -> None:
    def load(name):
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    cq, rq = load("capture_quality_report.json"), load("reconstruction_quality_report.json")
    meas, scaled = load("measurements.json"), load("measurements_cm.json")
    if cq:
        print(f"\nCapture quality: {cq['verdict']}")
        for k, c in cq["checks"].items():
            print(f"  {c['status']:>4}  {k}")
        for f in cq["feedback"]:
            print(f"   -> {f}")
    if rq:
        print(f"\nReconstruction quality: {rq['verdict']}")
        for k, c in rq["checks"].items():
            print(f"  {c['status']:>4}  {k}: {c.get('value') if not isinstance(c.get('value'), dict) else ''}")
    if meas:
        unit = "cm" if scaled else "x height"
        print(f"\nMeasurements ({unit}):")
        if scaled:
            print(f"  {'height':<16} {scaled['height_cm']:7.1f}  (supplied, not measured)")
        for k, m in meas["measurements"].items():
            v = (scaled["measurements"].get(k) or {}).get("tape_circumference_cm") if scaled else \
                m.get("tape_circumference_over_height")
            print(f"  {k:<16} {('%7.1f' if scaled else '%7.3f') % v if isinstance(v, (int, float)) else '   n/a'}")
        if not scaled:
            print("  (relative units: no height/calibration supplied; use --height-cm for centimetres)")
    t = (load("reconstruction_report.json") or {}).get("timings_sec", {})
    if t:
        print("\nTimings (s): " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("video", nargs="?", type=Path, help="360-degree recording (MP4/MOV)")
    src.add_argument("--video-id", help="an already uploaded recording id")
    ap.add_argument("--consent-attested", action="store_true",
                    help="operator confirms authorisation and the subject's consent (required for a new video)")
    ap.add_argument("--height-cm", type=float, help="subject's barefoot height; the only source of absolute scale")
    ap.add_argument("--person", type=int, help="candidate id when several people are visible")
    ap.add_argument("--shapy", action="store_true", help="also run the SHAPY comparison model")
    ap.add_argument("--baseline-run", help="earlier run id to compare against in the HTML report")
    args = ap.parse_args()

    from backend.services import body_reconstruction as br
    from backend.services import scan_report
    from backend.services.inference_engine import get_engine

    t0 = time.time()
    video_id = args.video_id or _ingest(args.video, args.consent_attested)
    engine = get_engine()
    engine._load()
    if not engine.is_ready:
        sys.exit(f"HMR2 not ready: {engine.load_error}")
    try:
        run = br.start_run(video_id, height_cm=args.height_cm)
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"Cannot start: {exc}")
    run_id = run["run_id"]
    run_dir = br._run_dir(run_id).resolve()
    r = _wait(br, run_id, args.person)
    if r["status"] != "completed":
        print(f"\nReconstruction {r['status']}: {(r.get('error') or {}).get('message', r.get('message'))}")
        _print_summary(run_dir, r)
        print(f"\noutputs: {run_dir}")
        sys.exit(2)

    if args.shapy:
        from backend.services import shapy_shape, video_processor
        if not shapy_shape.available():
            print("[shapy] model files not installed - skipped")
        else:
            print("[shapy] running SHAPY on the selected frames", flush=True)
            source = next(video_processor._OUTPUT_ROOT.joinpath(video_id).glob("original.*"))
            try:
                shapy_shape.compute_for_run(run_dir, source)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                print(f"[shapy] failed: {exc}")

    baseline = None
    if args.baseline_run:
        baseline = br._run_dir(args.baseline_run).resolve()
        if not (baseline / "reconstruction_report.json").is_file():
            print(f"[report] baseline run {args.baseline_run} has no report - comparison skipped")
            baseline = None
    page = scan_report.write_html(run_dir, baseline)
    _print_summary(run_dir, r)
    print(f"\nTotal wall time {time.time() - t0:.0f} s")
    print(f"mesh:    {run_dir / 'final_body_mesh.obj'}")
    print(f"report:  {page}")
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()
