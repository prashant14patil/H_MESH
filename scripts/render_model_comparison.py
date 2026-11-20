r"""Side-by-side images of the two measurement models: pipeline body vs SHAPY body, with measurement rings.

For each recording: pipeline body (shared SMPL shape in the measurement A-pose; betas from a run's fit_result.npz or
from a reliability-study baseline) and SHAPY's SMPL-X body (shapy_body_mesh.obj of a run of the same video). Both
are scaled to the given stature and drawn front / side / back at the same scale; each ring is re-cut from the
plane stored in the model's measurements file, so it is exactly the section that was measured. A table lists both
models' girths and, optionally, tape values.

Usage:
  .venv\Scripts\python.exe scripts/render_model_comparison.py --height-cm 179 \
      --case "B=study:outputs/accuracy_study/reliability_p02/9e126c35.json,shapy:9d237499" \
      --tape waist=88-90 abdomen=95-100 hip=98-103 --output outputs/model_comparison/B.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"
SITES = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "forearm_left", "thigh_left", "calf_left",
         "upper_arm_right", "forearm_right", "thigh_right", "calf_right"]
LABEL = {"chest": "Chest", "waist": "Waist", "abdomen": "Abdomen", "hip": "Hip", "upper_arm_left": "Upper arm L",
         "upper_arm_right": "Upper arm R", "forearm_left": "Forearm L", "forearm_right": "Forearm R",
         "thigh_left": "Thigh L", "thigh_right": "Thigh R", "calf_left": "Calf L", "calf_right": "Calf R"}
COLOR = {"chest": (60, 60, 220), "waist": (40, 160, 240), "abdomen": (40, 200, 200), "hip": (60, 170, 60),
         "upper_arm": (200, 120, 40), "forearm": (200, 60, 160), "thigh": (150, 80, 40), "calf": (120, 40, 200)}


def color(site):
    return COLOR["upper_arm" if site.startswith("upper_arm") else site.split("_")[0]]


def run_dir(prefix):
    return next(RUNS.glob(prefix + "*"))


def pipeline_body(spec, smpl):
    """spec: 'run:<id>' (fit_result.npz) or 'study:<json>' (baseline betas of a reliability study)."""
    kind, ref = spec.split(":", 1)
    if kind == "run":
        betas = np.load(run_dir(ref) / "diagnostics" / "fit_result.npz")["betas"]
        src = f"run {ref[:8]}"
    else:
        rep = json.loads(Path(ref).read_text(encoding="utf-8"))
        rid, d = next(iter(rep["runs"].items()))
        betas = np.asarray(d["conditions"]["baseline#0"]["betas"])
        prior = next((w for w in rep.get("fit", "").split(",") if "HMR2 prior" in w), "").strip()
        src = f"{prior or 'refit'}, refit of {rid}"
    v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, "cpu", pose="measurement")
    faces = np.asarray(smpl.faces, dtype=np.int64)
    meas = bm.measure_body(v, faces, smv.smpl_joints24(smpl, v))["measurements"]
    rings = {s: np.asarray(m["ring"]) for s, m in meas.items() if m.get("valid")}
    ratio = {s: m["tape_circumference_over_height"] for s, m in meas.items() if m.get("valid")}
    return v, faces, rings, ratio, src


def shapy_body(run_prefix):
    d = run_dir(run_prefix)
    v, f = smv.read_obj(d / "shapy_body_mesh.obj")
    rec = json.loads((d / "shapy_measurements.json").read_text(encoding="utf-8"))
    rings, ratio = {}, {}
    for s, m in rec["measurements"].items():
        if not m.get("valid"):
            continue
        sec = bm.measure_section(v, f, m["plane_origin"], m["plane_normal"])  # the plane that was measured
        if sec.get("points3d") is not None:
            rings[s] = np.asarray(sec["points3d"])
        ratio[s] = m["tape_circumference_over_height"]
    return v, np.asarray(f), rings, ratio, f"SHAPY, run {d.name[:8]} ({rec.get('frames_used')} frames)"


def render(v, faces, rings, yaw, scale, size_hw):
    """Flat-shaded orthographic view; rings drawn where they face the viewer."""
    H, W = size_hw
    a = math.radians(yaw)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    vr = v @ rot.T
    img = np.full((H, W, 3), 250, np.uint8)

    def px(p):
        return np.stack([W / 2 + p[:, 0] * scale, H - 20 - p[:, 1] * scale], 1)
    P = px(vr)
    tri = vr[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, 0.45, 0.82])
    for fi in np.argsort(tri[:, :, 2].mean(1)):
        if n[fi, 2] <= 0:
            continue
        s = 90 + 150 * max(float(n[fi] @ light), 0.0)
        cv2.fillConvexPoly(img, P[faces[fi]].astype(np.int32), (int(s * 0.93), int(s * 0.95), int(s)), cv2.LINE_AA)
    for site, r in rings.items():
        rr = r @ rot.T
        front = rr[:, 2] >= np.median(rr[:, 2]) - 1e-9
        q = px(rr).astype(np.int32)
        for i in range(len(q)):
            j = (i + 1) % len(q)
            if front[i] and front[j]:
                cv2.line(img, tuple(q[i]), tuple(q[j]), color(site), 3, cv2.LINE_AA)
    return img


def panel(title, v, faces, rings, size_h, scale):
    views = [render(v, faces, rings, yaw, scale, (size_h, int(size_h * 0.64))) for yaw in (0, 90, 180)]
    strip = np.hstack(views)
    head = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    for k, lab in enumerate(("front", "side", "back")):
        cv2.putText(strip, lab, (k * views[0].shape[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1,
                    cv2.LINE_AA)
    return np.vstack([head, strip])


def table(rows, tape, height_cm, h):
    w = 520
    img = np.full((h, w, 3), 255, np.uint8)
    y = 36
    cv2.putText(img, f"Girths (cm) at {height_cm:g} cm", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2,
                cv2.LINE_AA)
    y += 34
    for x, t in ((12, "Site"), (200, "Pipeline"), (300, "SHAPY"), (400, "Tape")):
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    y += 10
    cv2.line(img, (10, y), (w - 10, y), (200, 200, 200), 1)
    for site, p, s in rows:
        y += 30
        cv2.rectangle(img, (12, y - 14), (26, y), color(site), -1)
        cv2.putText(img, LABEL[site], (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
        for x, val in ((200, p), (300, s)):
            cv2.putText(img, "-" if val is None else f"{val:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (30, 30, 30), 1, cv2.LINE_AA)
        if site in tape:
            cv2.putText(img, tape[site], (400, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 110, 0), 2, cv2.LINE_AA)
    notes = ["Both bodies scaled to the same stature;",
             "rings are the exact measured sections.",
             "Tape: self-measured ranges (20 Nov), loose",
             "clothing; neither model is ground truth.",
             "Not a body-composition measurement."]
    y = h - 20 - 24 * (len(notes) - 1)
    for line in notes:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        y += 24
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True,
                    help='"NAME=<run:ID|study:JSON>,shapy:RUN_ID" (repeatable; one image row each)')
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--tape", nargs="*", default=[], help="site=value or site=lo-hi")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from measurement_accuracy_study import load_smpl
    smpl = load_smpl("cpu")
    tape = dict(t.split("=", 1) for t in a.tape)
    rows_img, summary = [], {}
    for case in a.case:
        name, specs = case.split("=", 1)
        pipe_spec, shapy_spec = specs.split(",")
        pv, pf, pr, pratio, psrc = pipeline_body(pipe_spec, smpl)
        sv, sf, sr, sratio, ssrc = shapy_body(shapy_spec.split(":", 1)[1])
        k_p = a.height_cm / float(np.ptp(pv[:, 1]))
        k_s = a.height_cm / float(np.ptp(sv[:, 1]))
        pv, sv = pv * k_p, sv * k_s
        pr = {s: r * k_p for s, r in pr.items()}
        sr = {s: r * k_s for s, r in sr.items()}
        size_h = 620
        scale = (size_h - 40) / a.height_cm
        left = panel(f"{name}: Pipeline ({psrc})"[:70], pv, pf, pr, size_h, scale)
        right = panel(f"{name}: {ssrc}"[:70], sv, sf, sr, size_h, scale)
        rows = [(s, pratio.get(s) and pratio[s] * a.height_cm, sratio.get(s) and sratio[s] * a.height_cm)
                for s in SITES]
        tab = table(rows, tape, a.height_cm, left.shape[0])
        sep = np.full((left.shape[0], 12, 3), 230, np.uint8)
        rows_img.append(np.hstack([left, sep, right, sep, tab]))
        summary[name] = {"pipeline_source": psrc, "shapy_source": ssrc,
                         "girths_cm": {s: {"pipeline": p and round(p, 1), "shapy": q and round(q, 1)} for s, p, q in rows}}
        print(name, " ".join(f"{s}: {p:.1f}/{q:.1f}" for s, p, q in rows[:4] if p and q))
    width = max(r.shape[1] for r in rows_img)
    out = np.vstack([np.pad(r, ((0, 16), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows_img])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.output), out)
    a.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
