"""Item #4: is the subject mask systematically too wide or too narrow at the true image edge?

For every selected view of a run, the original frame is re-read from the video. Along the mask outline, an
intensity profile is sampled perpendicular to the boundary (from 8 px inside to 8 px outside), and the position of
the strongest image gradient is taken as the true edge. offset > 0 means the true edge lies OUTSIDE the mask (mask
too small); offset < 0 means the mask spills past the edge. Points on low-contrast boundaries are ignored.

Reported per body region (head / torso / arms / legs) because hair, skin and fabric segment differently.

    .venv\\Scripts\\python.exe scripts/mask_edge_bias.py RUN_ID [--height-cm 179]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import shapy_shape  # noqa: E402

HALF = 8          # profile half-length (px)
MIN_CONTRAST = 12.0  # minimum gradient (grey levels / px) for a boundary point to count


def regions(pts, kp):
    sh_y = kp[[5, 6], 1].mean(); hip_y = kp[[11, 12], 1].mean(); ank_y = kp[[15, 16], 1].max()
    x_l, x_r = sorted([kp[11, 0], kp[12, 0]])
    tor_l, tor_r = sorted([kp[5, 0], kp[6, 0]])
    lab = np.full(len(pts), "torso", dtype=object)
    lab[pts[:, 1] < sh_y] = "head_shoulders"
    lab[(pts[:, 1] > hip_y) & (pts[:, 1] <= ank_y)] = "legs"
    lab[pts[:, 1] > ank_y] = "feet"
    mid = (pts[:, 1] >= sh_y) & (pts[:, 1] <= hip_y)
    lab[mid & ((pts[:, 0] < tor_l - 0.15 * (tor_r - tor_l)) | (pts[:, 0] > tor_r + 0.15 * (tor_r - tor_l)))] = "arms"
    return lab


def edge_offsets(gray, mask):
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=len)[:, 0, :].astype(np.float64)
    # outward normal from the contour tangent (contour is CCW/CW; fix sign by testing a point)
    t = np.roll(c, -3, 0) - np.roll(c, 3, 0)
    n = np.stack([t[:, 1], -t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    probe = np.round(c + 4 * n).astype(int)
    probe[:, 0] = probe[:, 0].clip(0, mask.shape[1] - 1); probe[:, 1] = probe[:, 1].clip(0, mask.shape[0] - 1)
    if mask[probe[:, 1], probe[:, 0]].mean() > 0.5:
        n = -n
    s = np.arange(-HALF, HALF + 1, dtype=np.float64)
    xs = c[:, None, 0] + s[None] * n[:, None, 0]
    ys = c[:, None, 1] + s[None] * n[:, None, 1]
    prof = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    g = np.abs(np.diff(prof, axis=1))                       # gradient between consecutive samples
    k = g.argmax(1)
    off = (s[:-1] + 0.5)[k]                                 # + = edge outside the mask boundary
    strong = g.max(1) >= MIN_CONTRAST
    return c, off, strong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, default=179.0)
    a = ap.parse_args()
    d = next((ROOT / "outputs" / "reconstructions").glob(a.run_id + "*"))
    z = np.load(d / "diagnostics" / "fit_inputs.npz")
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    sel = json.loads((d / "selected_frames.json").read_text(encoding="utf-8"))["selected"]
    src = {s["frame_position"]: s["source_frame_index"] for s in sel}
    video = next((ROOT / "data" / "captures" / status["video_id"]).glob("original.*"))
    shape = tuple(int(x) for x in z["mask_shape"])
    frames = shapy_shape.read_display_frames(video, [src[int(p)] for p in z["selected"]], shape)
    per_region: dict[str, list] = {}
    per_view = []
    for i, pos in enumerate(z["selected"]):
        if not z["mask_usable"][i]:
            continue
        mask = np.unpackbits(z["masks_packed"][i])[:shape[0] * shape[1]].reshape(shape)
        frame = frames.get(src[int(pos)])
        if frame is None:
            continue
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
        c, off, strong = edge_offsets(gray, mask)
        lab = regions(c, z["keypoints"][i])
        px_per_cm = float(z["box_height"][i]) / a.height_cm
        for r in np.unique(lab):
            sel_r = strong & (lab == r)
            per_region.setdefault(r, []).extend((off[sel_r] / px_per_cm).tolist())
        per_view.append({"yaw": float(z["yaw"][i]), "median_offset_px": float(np.median(off[strong])),
                         "strong_fraction": float(strong.mean())})
    print(f"run {d.name[:8]}: true image edge relative to the mask boundary (+ = mask too SMALL, - = mask spills over)")
    print(f"{'region':16s} {'points':>7s} {'median cm':>10s} {'mean cm':>8s} {'IQR cm':>14s}")
    summary = {}
    for r, v in sorted(per_region.items()):
        v = np.asarray(v)
        q1, q3 = np.percentile(v, [25, 75])
        summary[r] = {"points": int(len(v)), "median_cm": float(np.median(v)), "mean_cm": float(v.mean()),
                      "iqr_cm": [float(q1), float(q3)]}
        print(f"{r:16s} {len(v):7d} {np.median(v):+10.2f} {v.mean():+8.2f}   [{q1:+.2f}, {q3:+.2f}]")
    allv = np.concatenate([np.asarray(v) for v in per_region.values()])
    print(f"overall median {np.median(allv):+.2f} cm per side -> girth effect ~ {2 * np.pi * np.median(allv):+.1f} cm "
          "if uniform around a torso section")
    out = d / "diagnostics" / "mask_edge_bias.json"
    out.write_text(json.dumps({"regions": summary, "views": per_view}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
