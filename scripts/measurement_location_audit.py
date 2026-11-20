r"""Where do the measurement rings land on the body? (anatomical-location audit, CPU only)

For each run (its fitted shared shape, re-posed in the measurement A-pose) or a synthetic SMPL shape, report per site:

* ring height as a fraction of stature (torso) / position along the bone (limbs);
* whether the min/max search ended on the edge of its range (the true extremum then lies OUTSIDE the
  anatomical zone, so the reading is a boundary value, not the intended landmark);
* cross-section width, depth and depth/width ratio, and the hull/contour concavity;
* thigh: the ring position versus the tape protocol (halfway between the crotch, found on the mesh, and
  the top of the kneecap), and the girth difference between the two locations.

Usage:
  .venv\\Scripts\\python.exe scripts/measurement_location_audit.py RUN_ID [RUN_ID ...] [--synthetic] --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"
# Top of the kneecap above the SMPL knee joint centre (patella top ~4-5 cm above the joint line in adults):
# an approximation, so the thigh comparison is also given as girth change per % of stature.
PATELLA_TOP_ABOVE_KNEE_FRAC = 0.025


def body(smpl, betas):
    v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, "cpu", pose="measurement")
    return v, smv.smpl_joints24(smpl, v)


def crotch_height(v, faces, J):
    """Lowest torso plane whose loop still encloses the body axis (below it the plane cuts two separate legs)."""
    axis = J[[bm.PELVIS, bm.SPINE1, bm.SPINE2, bm.SPINE3]][:, [0, 2]].mean(0)

    def joined(y):
        p = np.array([axis[0], y, axis[1]])
        return bm.measure_section(v, faces, p, [0, 1, 0], p)["enclosing_loops"] > 0
    lo, hi = float(J[[bm.L_KNEE, bm.R_KNEE], 1].mean()), float(J[bm.PELVIS, 1])
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if joined(mid) else (mid, hi)
    return hi


def thigh_at(v, faces, J, hip, knee, frac, other):
    p = J[hip] + frac * (J[knee] - J[hip])
    m = bm.measure_section(v, faces, p, J[knee] - J[hip], p, other)
    return m.get("tape_perimeter") if m.get("valid") else None


SEARCH = {  # mirrors body_measurements.measure_body: (kind, number of candidate planes)
    "chest": 25, "waist": 25, "hip": 25, "forearm": 12, "calf": 12}


def audit(smpl, betas, faces):
    v, J = body(smpl, betas)
    H = float(v[:, 1].max() - v[:, 1].min())
    floor = float(v[:, 1].min())
    meas = bm.measure_body(v, faces, J)["measurements"]
    knee_y = float(J[[bm.L_KNEE, bm.R_KNEE], 1].mean())
    y = {k: float(J[i, 1]) for k, i in (("pelvis", bm.PELVIS), ("spine1", bm.SPINE1), ("spine3", bm.SPINE3))}
    shoulder_y = float(J[[bm.L_SHOULDER, bm.R_SHOULDER], 1].mean())
    ranges = {"chest": (y["spine3"], y["spine3"] + 0.5 * (shoulder_y - y["spine3"])),
              "waist": (y["spine1"], y["spine3"]), "hip": (y["pelvis"] - 0.4 * (y["pelvis"] - knee_y), y["pelvis"]),
              "forearm": (0.12, 0.45), "calf": (0.15, 0.5)}
    out = {"stature_mesh_units": H, "landmarks_frac_of_stature": {
        "pelvis_joint": (y["pelvis"] - floor) / H, "spine1_joint": (y["spine1"] - floor) / H,
        "spine3_joint": (y["spine3"] - floor) / H, "shoulder_joints": (shoulder_y - floor) / H,
        "knee_joints": (knee_y - floor) / H}, "sites": {}}
    crotch = crotch_height(v, faces, J)
    out["landmarks_frac_of_stature"]["crotch_mesh"] = (crotch - floor) / H
    for name, m in meas.items():
        if not m["valid"]:
            out["sites"][name] = {"valid": False}
            continue
        ring = np.asarray(m["ring"])
        n = np.asarray(m["plane_normal"])
        _, u, w = bm._plane_basis(n)
        rel = ring - ring.mean(0)
        ext_u, ext_w = float(np.ptp(rel @ u)), float(np.ptp(rel @ w))
        row = {"valid": True, "girth_over_stature": m["tape_circumference_over_height"],
               "ring_centre_height_frac": float((ring[:, 1].mean() - floor) / H),
               "concavity_ratio": m["concavity_ratio"]}
        if name in ("chest", "waist", "abdomen", "hip"):  # horizontal plane: x = width, z = depth
            width, depth = float(np.ptp(ring[:, 0])), float(np.ptp(ring[:, 2]))
            row.update(width_over_stature=width / H, depth_over_stature=depth / H, depth_over_width=depth / width)
        else:
            row.update(section_major_minor_ratio=max(ext_u, ext_w) / max(min(ext_u, ext_w), 1e-9))
        base = name.split("_")[0] if not name.startswith("upper_arm") else "upper_arm"
        if base in ranges:
            lo, hi = ranges[base]
            t = m["search_parameter"]
            span = hi - lo
            row["search_position_in_range"] = float((t - lo) / span) if span else None
            row["extremum_at_range_edge"] = bool(abs(t - lo) < 1e-9 or abs(t - hi) < 1e-9)
        out["sites"][name] = row
    # Thigh: pipeline (hip-knee joint midpoint) vs tape protocol (midway crotch -> top of kneecap).
    for side, (hp, kn, other) in {"left": (bm.L_HIP, bm.L_KNEE, [(J[bm.R_HIP], J[bm.R_KNEE])]),
                                  "right": (bm.R_HIP, bm.R_KNEE, [(J[bm.L_HIP], J[bm.L_KNEE])])}.items():
        patella_top = J[kn, 1] + PATELLA_TOP_ABOVE_KNEE_FRAC * H
        target_y = (crotch + patella_top) / 2
        frac = float((J[hp, 1] - target_y) / (J[hp, 1] - J[kn, 1]))
        g_pipe = thigh_at(v, faces, J, hp, kn, 0.5, other)
        g_prot = thigh_at(v, faces, J, hp, kn, frac, other)
        g_lo, g_hi = thigh_at(v, faces, J, hp, kn, frac - 0.05, other), thigh_at(v, faces, J, hp, kn, frac + 0.05, other)
        bone = float(np.linalg.norm(J[hp] - J[kn]))
        out["sites"][f"thigh_{side}"]["protocol_check"] = {
            "pipeline_bone_fraction": 0.5, "protocol_bone_fraction": frac,
            "pipeline_height_frac": float((J[hp, 1] + 0.5 * (J[kn, 1] - J[hp, 1]) - floor) / H),
            "protocol_height_frac": float((target_y - floor) / H),
            "pipeline_girth_over_stature": g_pipe / H if g_pipe else None,
            "protocol_girth_over_stature": g_prot / H if g_prot else None,
            "pipeline_minus_protocol_pct": 100 * (g_pipe - g_prot) / g_prot if g_pipe and g_prot else None,
            "girth_change_pct_per_1pct_stature_along_thigh":
                (100 * (g_lo - g_hi) / g_prot) / (100 * 0.10 * bone / H) if g_lo and g_hi and g_prot else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--synthetic", action="store_true", help="also audit SMPL mean and +-2 SD shapes 1-2")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from measurement_accuracy_study import load_smpl
    smpl = load_smpl("cpu")
    faces = np.asarray(smpl.faces, dtype=np.int64)
    cases = {}
    for run in a.runs:
        d = next(RUNS.glob(run + "*"))
        cases[d.name[:8]] = np.load(d / "diagnostics" / "fit_result.npz")["betas"]
    if a.synthetic:
        cases["smpl_mean"] = np.zeros(10)
        for k in (0, 1):
            for s in (-2, 2):
                b = np.zeros(10)
                b[k] = s
                cases[f"beta{k + 1}_{s:+d}sd"] = b
    rep = {}
    for name, b in cases.items():
        rep[name] = audit(smpl, np.asarray(b, dtype=np.float64), faces)
        s = rep[name]["sites"]
        print(name, " ".join(f"{k}:{s[k]['ring_centre_height_frac']:.3f}{'!' if s[k].get('extremum_at_range_edge') else ''}"
                             for k in ("chest", "waist", "abdomen", "hip") if s[k]["valid"]),
              "thigh pipe-prot %:", round(s["thigh_left"]["protocol_check"]["pipeline_minus_protocol_pct"] or 0, 2), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
