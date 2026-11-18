r"""End-to-end check of the legacy quality -> subject review -> per-frame mesh workflow (real API, real HMR2).

    .venv\Scripts\python.exe scripts/validate_review_workflow.py --video <recording id> --report <quality report id> \
        --confirm frame_001.jpg frame_005.jpg ... [--reject frame_002.jpg] [--experimental]

The frames passed to --confirm must have been inspected by the operator running this script: the
script records a reviewer confirmation for each of them (stored as a normal review decision in a
new session). Existing recording authorization is required and is never created here.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

B = "http://127.0.0.1:8000"
CHECKS = ("readability", "person_presence", "sharpness", "lighting", "full_body")


def call(method, path, body=None, timeout=120):
    req = urllib.request.Request(B + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def subject_box(frame):
    """Union of the pose landmark extent and the matching detector box, padded slightly."""
    lb = frame["landmark_bbox_normalized"]
    boxes = [lb] + [d["box"] for d in frame.get("detector_evidence") or []]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return [max(0.0, x0 - 0.01), max(0.0, y0 - 0.01), min(1.0, x1 + 0.01), min(1.0, y1 + 0.01)]


def wait_run(run_id):
    while True:
        s, run = call("GET", f"/api/video-mesh/runs/{run_id}")
        if run.get("status") != "processing":
            return run
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--confirm", nargs="+", required=True)
    ap.add_argument("--reject", nargs="*", default=[])
    ap.add_argument("--experimental", action="store_true")
    a = ap.parse_args()
    results = {}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        results[name] = bool(cond)
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    s, report = call("GET", f"/api/capture-quality/runs/{a.report}/report")
    check("quality report retrieved", s == 200 and report.get("video_id") == a.video, report.get("detector_method"))
    frames = {f["filename"]: f for f in report["frames"]}
    eligible = [n for n, f in frames.items() if all(f["checks"][k] == "pass" for k in CHECKS)]
    check("frames with all mandatory checks passing", len(eligible) > 0, f"{len(eligible)}/{len(frames)}")

    ref = a.confirm[0]
    s, session = call("POST", "/api/subject-review/sessions", {
        "video_id": a.video, "quality_run_id": a.report, "reference_frame": ref, "bbox": subject_box(frames[ref])})
    check("reference subject + review session created", s == 201, session.get("detail", session.get("session_id", "")))
    sid = session["session_id"]

    for name in a.confirm:
        s, d = call("POST", f"/api/subject-review/sessions/{sid}/decisions",
                    {"filename": name, "decision": "confirm", "bbox": subject_box(frames[name])})
        check(f"confirm {name}", s == 200 and d.get("identity_status") == "human_confirmed_not_biometric", d.get("detail", ""))
    for name in a.reject:
        s, d = call("POST", f"/api/subject-review/sessions/{sid}/decisions", {"filename": name, "decision": "reject"})
        check(f"reject {name}", s == 200 and d.get("reviewer_decision") == "reject", d.get("detail", ""))

    s, got = call("GET", f"/api/subject-review/sessions/{sid}/decisions/{a.confirm[-1]}")
    check("decision persisted (GET)", s == 200 and got.get("reviewer_decision") == "confirm")
    s, resumed = call("GET", f"/api/subject-review/sessions/{sid}/resume?video_id={a.video}&quality_run_id={a.report}")
    check("session resumed with all decisions", s == 200 and set(resumed.get("decisions", {})) == set(a.confirm) | set(a.reject),
          f"{len(resumed.get('decisions', {}))} decisions, unavailable={resumed.get('unavailable_frames')}")

    def ready(names, reviewed, session_id=sid):
        return call("POST", "/api/video-mesh/readiness", {"video_id": a.video, "frames": names, "quality_run_id": a.report,
                                                         "reviewed_frames": reviewed, "review_session_id": session_id})[1]

    sel = a.confirm[:12]
    unreviewed = next((n for n in eligible if n not in a.confirm and n not in a.reject), None)
    negatives = {
        "no frames selected": ready([], []),
        "13 frames selected": ready(sorted(frames)[:13], sorted(frames)[:13]),
        "missing warning acknowledgement": ready(sel, sel[:-1]),
        "wrong review session": ready(sel, sel, "00000000-0000-4000-8000-000000000000"),
    }
    if a.reject:
        negatives["rejected frame selected"] = ready(sel[:1] + a.reject[:1], sel[:1] + a.reject[:1])
    if unreviewed:
        negatives["frame without subject review"] = ready(sel[:1] + [unreviewed], sel[:1] + [unreviewed])
    for name, r in negatives.items():
        check(f"blocked: {name}", r.get("ready") is False, "; ".join(r.get("blockers", []))[:160])

    r = ready(sel, sel)
    check("valid reviewed selection accepted by readiness", r.get("ready") is True, "; ".join(r.get("blockers", [])))
    body = {"video_id": a.video, "frames": sel, "quality_run_id": a.report, "reviewed_frames": sel, "review_session_id": sid}
    s, d = call("POST", "/api/video-mesh/runs", {**body, "final_confirmation": False})
    check("blocked: start without final confirmation", s == 400, d.get("detail", ""))
    s, run = call("POST", "/api/video-mesh/runs", {**body, "final_confirmation": True})
    check("reviewed HMR2 batch started", s == 202, run.get("detail", run.get("run_id", "")))
    if s == 202:
        run = wait_run(run["run_id"])
        done = [f for f in run["frames"] if f["status"] == "completed"]
        check("reviewed batch completed with meshes", run["status"] in ("completed", "completed_with_errors") and done,
              f"status={run['status']} meshes={len(done)}/{len(run['frames'])}")
        for f in done[:2]:
            req = urllib.request.urlopen(f"{B}/api/video-mesh/runs/{run['run_id']}/mesh/{f['filename']}", timeout=60)
            text = req.read().decode()
            v = sum(1 for line in text.splitlines() if line.startswith("v "))
            fc = sum(1 for line in text.splitlines() if line.startswith("f "))
            check(f"mesh file for {f['filename']}", v == 6890 and fc == 13776, f"{v} vertices, {fc} faces")
        results["reviewed_run_id"] = run["run_id"]

    if a.experimental:
        s, d = call("POST", "/api/video-mesh/experimental/runs", {"video_id": a.video, "quality_run_id": a.report})
        check("blocked: experimental without acknowledgement", s == 400, d.get("detail", ""))
        s, run = call("POST", "/api/video-mesh/experimental/runs", {
            "video_id": a.video, "quality_run_id": a.report, "experimental_risk_acknowledged": True, "final_confirmation": True})
        check("experimental whole-frame run started", s == 202, run.get("detail", run.get("run_id", "")))
        if s == 202:
            run = wait_run(run["run_id"])
            done = [f for f in run["frames"] if f["status"] == "completed"]
            check("experimental run produced meshes", bool(done), f"status={run['status']} meshes={len(done)}/{len(run['frames'])}")
            results["experimental_run_id"] = run["run_id"]

    results["session_id"] = sid
    print(json.dumps(results, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
