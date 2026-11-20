"""Plausibility of a reconstruction's torso girths against ANSUR II (US Army 2012 survey, public 2017 release).

ANSUR II tape-measured 4,082 men and 1,986 women in minimal clothing. In real people chest, navel-level waist and
buttock girths are strongly linked; each is predicted here from the other two plus stature (sex-specific linear
regression; residual SD = how much real people vary around that prediction). A reconstructed girth more than
``Z_FLAG`` SD from its prediction is a combination real people rarely have - typically a local error such as loose
trousers inflating the hip. It flags WHICH girth is out of line; it does not say which value is right.

Pipeline site -> ANSUR measurement: chest -> chestcircumference (nipple level), abdomen -> waistcircumference
(ANSUR's waist is at the navel), hip -> buttockcircumference. Data: data/reference/ansur2/ANSUR_II_*_Public.csv
(https://www.openlab.psu.edu/ansur2/), millimetres.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "reference" / "ansur2"
FILES = {"male": "ANSUR_II_MALE_Public.csv", "female": "ANSUR_II_FEMALE_Public.csv"}
SITES = {"chest": "chestcircumference", "abdomen": "waistcircumference", "hip": "buttockcircumference"}
Z_FLAG = 2.0


def available(sex: str | None) -> bool:
    return sex in FILES and (DATA / FILES[sex]).is_file()


@lru_cache(maxsize=2)
def _models(sex: str) -> dict:
    rows = list(csv.DictReader(open(DATA / FILES[sex], encoding="latin-1")))
    col = {c: np.array([float(r[c]) for r in rows]) for c in list(SITES.values()) + ["stature"]}
    models = {}
    for site, c in SITES.items():
        others = [s for s in SITES if s != site]
        X = np.stack([col[SITES[o]] for o in others] + [col["stature"], np.ones(len(rows))], 1)
        coef, *_ = np.linalg.lstsq(X, col[c], rcond=None)
        models[site] = (others, coef, float((col[c] - X @ coef).std()))
    return {"n": len(rows), "models": models}


def _ring_edges(vertices, faces, origin, normal, loop_points):
    """(a, b, t) per ring point: the mesh edge (a, b) each section point lies on and its position t along it."""
    v = np.asarray(vertices, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64); n = n / np.linalg.norm(n)
    d = (v - np.asarray(origin, dtype=np.float64)) @ n
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    e = e[(d[e[:, 0]] > 0) != (d[e[:, 1]] > 0)]
    t = d[e[:, 0]] / (d[e[:, 0]] - d[e[:, 1]])
    pts = v[e[:, 0]] + t[:, None] * (v[e[:, 1]] - v[e[:, 0]])
    idx = np.array([int(np.argmin(np.linalg.norm(pts - p, axis=1))) for p in np.asarray(loop_points)])
    return e[idx, 0], e[idx, 1], t[idx]


def build_girth_prior(smpl, betas, sex: str, stature_cm: float, weight: float, device="cpu"):
    """Population-consistency term for the shape fit (see smpl_multiview.GirthPrior), or None if unavailable.

    Rings are the chest / navel-waist / hip sections of the measurement A-pose body for ``betas`` (the fit's
    starting shape), stored as fixed mesh edges so their length is differentiable in the shape. ``calib`` converts
    ring length to tape (convex) length at the start shape."""
    if not available(sex):
        return None
    from backend.services import body_measurements as bm
    from backend.services import smpl_multiview as smv
    v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, device, pose="measurement")
    faces = np.asarray(smpl.faces, dtype=np.int64)
    meas = bm.measure_body(v, faces, smv.smpl_joints24(smpl, v))["measurements"]
    rings = {}
    for site in SITES:
        m = meas.get(site, {})
        if not m.get("valid"):
            return None
        sec = bm.measure_section(v, faces, np.array(m["plane_origin"]), np.array(m["plane_normal"]))
        a, b, t = _ring_edges(v, faces, m["plane_origin"], m["plane_normal"], sec["points3d"])
        loop = np.asarray(sec["points3d"])
        loop_len = float(np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1).sum())
        rings[site] = (a, b, t, sec["tape_perimeter"] / loop_len)
    return smv.GirthPrior(rings=rings, models=_models(sex)["models"], stature_mm=float(stature_cm) * 10.0,
                          weight=float(weight))


def check(girths_cm: dict, stature_cm: float, sex: str) -> dict | None:
    """{"sites": {site: {mesh_cm, typical_cm, sd_cm, z, out_of_line}}, "n_reference": n} or None if unavailable."""
    if not available(sex) or any(girths_cm.get(s) is None for s in SITES):
        return None
    m = _models(sex)
    sites = {}
    for site, (others, coef, sd) in m["models"].items():
        pred = float(np.array([girths_cm[o] * 10 for o in others] + [stature_cm * 10, 1.0]) @ coef)
        z = (girths_cm[site] * 10 - pred) / sd
        sites[site] = {"mesh_cm": round(float(girths_cm[site]), 1), "typical_cm": round(pred / 10, 1),
                       "sd_cm": round(sd / 10, 1), "z": round(float(z), 2), "out_of_line": bool(abs(z) > Z_FLAG)}
    return {"sites": sites, "n_reference": m["n"], "sex": sex, "z_flag": Z_FLAG,
            "reference": "ANSUR II public data (US Army 2012 anthropometric survey)",
            "note": "Typical value of each girth for people with the same other two girths and stature. "
                    "Out of line = rare combination (e.g. loose trousers inflating the hip); not a tape measurement."}
