"""Circumferences measured on the final 3D mesh by plane sections.

Every circumference is computed from the mesh geometry only (never from image widths):

1. anatomical reference from SMPL's native joints (J_regressor on the final vertices);
2. a cutting plane (horizontal for the torso, perpendicular to the bone for limbs);
3. exact mesh-plane intersection assembled into closed loops edge-by-edge;
4. the loop that encloses the anatomical reference point is selected;
5. perimeter of that loop and of its convex hull (what a tape measure follows across
   concavities, e.g. between the buttocks);
6. validity checks (closed, encloses the reference, single enclosing loop).

Units are the mesh's units. Without camera calibration or a known-size reference the mesh
scale is RELATIVE, so values are also reported as a fraction of mesh height. They are not
centimetres and are not body-composition measurements.
"""
from __future__ import annotations

import math

import numpy as np

# SMPL native joint indices.
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
NECK, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 12, 16, 17, 18, 19, 20, 21


def _plane_basis(normal):
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)


def section_loops(vertices: np.ndarray, faces: np.ndarray, origin, normal) -> list[dict]:
    """Closed and open polylines where the plane cuts the mesh.

    Intersection points are computed once per mesh edge, so neighbouring faces share
    exactly the same point and loops close without tolerance matching.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n, u, w = _plane_basis(normal)
    d = (v - np.asarray(origin, dtype=np.float64)) @ n
    eps = 1e-9
    d = np.where(np.abs(d) < eps, eps, d)  # a vertex exactly on the plane is nudged to one side
    side = d > 0
    edges = {}
    neighbours: dict[tuple, list] = {}
    for tri in f:
        crossing = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if side[a] != side[b]:
                key = (a, b) if a < b else (b, a)
                if key not in edges:
                    t = d[key[0]] / (d[key[0]] - d[key[1]])
                    edges[key] = v[key[0]] + t * (v[key[1]] - v[key[0]])
                crossing.append(key)
        if len(crossing) == 2:
            neighbours.setdefault(crossing[0], []).append(crossing[1])
            neighbours.setdefault(crossing[1], []).append(crossing[0])
    loops, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        path, prev, cur, closed = [start], None, start, False
        seen.add(start)
        while True:
            nxt = [e for e in neighbours[cur] if e != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                closed = True
                break
            if cur in seen:
                break
            seen.add(cur)
            path.append(cur)
        pts3 = np.array([edges[e] for e in path])
        rel = pts3 - np.asarray(origin, dtype=np.float64)
        pts2 = np.stack([rel @ u, rel @ w], 1)
        loops.append({"points3d": pts3, "points2d": pts2, "closed": closed})
    return loops


def _perimeter(pts2, closed=True):
    seg = np.diff(np.vstack([pts2, pts2[:1]]) if closed else pts2, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def _hull(pts2):
    pts = sorted(map(tuple, np.round(pts2, 12)))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _contains(pts2, q):
    x, y = q
    inside = False
    n = len(pts2)
    for i in range(n):
        x0, y0 = pts2[i]
        x1, y1 = pts2[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _segment_plane_point(a, b, origin, n):
    da, db = (np.asarray(a) - origin) @ n, (np.asarray(b) - origin) @ n
    if da * db > 0 or da == db:
        return None
    t = da / (da - db)
    return np.asarray(a) + t * (np.asarray(b) - np.asarray(a))


def measure_section(vertices, faces, origin, normal, reference=None, exclude_segments=()) -> dict:
    """Circumference of the closed loop enclosing ``reference`` (default: the plane origin).

    ``exclude_segments`` are bone segments (pairs of 3D joints) of OTHER body parts: if one
    crosses the plane inside the selected loop, that loop has merged with another body part
    (e.g. an arm touching the torso) and the section is invalid.
    """
    origin = np.asarray(origin, dtype=np.float64)
    ref = origin if reference is None else np.asarray(reference, dtype=np.float64)
    loops = section_loops(vertices, faces, origin, normal)
    n, u, w = _plane_basis(normal)
    q = np.array([(ref - origin) @ u, (ref - origin) @ w])
    enclosing = [lp for lp in loops if lp["closed"] and len(lp["points2d"]) >= 3 and _contains(lp["points2d"], q)]
    result = {"loops_in_plane": len(loops), "closed_loops": sum(lp["closed"] for lp in loops),
              "enclosing_loops": len(enclosing), "plane_origin": origin.tolist(), "plane_normal": n.tolist()}
    if not enclosing:
        result.update(valid=False, reason="no closed loop encloses the anatomical reference point")
        return result
    # Nested loops (e.g. an inner surface) would also enclose the point; the outermost is the skin.
    loop = max(enclosing, key=lambda lp: _perimeter(lp["points2d"]))
    hull = _hull(loop["points2d"])
    perimeter, hull_perimeter = _perimeter(loop["points2d"]), _perimeter(hull)
    merged = []
    for a, b in exclude_segments:
        p = _segment_plane_point(a, b, origin, n)
        if p is not None and _contains(loop["points2d"], np.array([(p - origin) @ u, (p - origin) @ w])):
            merged.append(True)
    result.update(valid=len(enclosing) == 1 and not merged, perimeter=perimeter, tape_perimeter=hull_perimeter,
                  concavity_ratio=perimeter / hull_perimeter if hull_perimeter else None,
                  points3d=loop["points3d"], n_points=len(loop["points3d"]))
    if len(enclosing) != 1:
        result["reason"] = "more than one closed loop encloses the reference point"
    elif merged:
        result["reason"] = "section merges with another body part"
    return result


def _search(vertices, faces, plane_fn, ts, pick, exclude=()):
    """Evaluate planes along a parameter range and keep the min/max valid tape perimeter."""
    best = None
    for t in ts:
        origin, normal, ref = plane_fn(t)
        m = measure_section(vertices, faces, origin, normal, ref, exclude)
        if not m.get("valid"):
            continue
        if best is None:
            best = (t, m)
        elif pick == "max" and m["tape_perimeter"] > best[1]["tape_perimeter"]:
            best = (t, m)
        elif pick == "min" and m["tape_perimeter"] < best[1]["tape_perimeter"]:
            best = (t, m)
    return best


def measure_body(vertices: np.ndarray, faces: np.ndarray, joints24: np.ndarray) -> dict:
    """Circumferences on an upright (y-up, +z front) mesh in a pose with arms clear of the torso."""
    v = np.asarray(vertices, dtype=np.float64)
    J = np.asarray(joints24, dtype=np.float64)
    height = float(v[:, 1].max() - v[:, 1].min())
    up = np.array([0.0, 1.0, 0.0])
    axis_xz = J[[PELVIS, SPINE1, SPINE2, SPINE3], :][:, [0, 2]].mean(0)

    def torso(y):
        p = np.array([axis_xz[0], y, axis_xz[1]])
        return p, up, p

    def limb(a, b, frac):
        p = J[a] + frac * (J[b] - J[a])
        return p, J[b] - J[a], p

    shoulder_y = float(J[[L_SHOULDER, R_SHOULDER], 1].mean())
    knee_y = float(J[[L_KNEE, R_KNEE], 1].mean())
    y = {k: float(J[i, 1]) for k, i in (("pelvis", PELVIS), ("spine1", SPINE1), ("spine2", SPINE2), ("spine3", SPINE3))}
    steps = 25
    specs = {
        "chest": ("torso", "max", np.linspace(y["spine3"], y["spine3"] + 0.5 * (shoulder_y - y["spine3"]), steps),
                  "maximum torso girth between the SMPL spine3 joint and halfway to the shoulder joints (below the armpits)"),
        "waist": ("torso", "min", np.linspace(y["spine1"], y["spine3"], steps),
                  "minimum torso girth between the spine1 and spine3 joints (natural waist)"),
        "abdomen": ("torso", "max", [y["spine1"]],
                    "torso girth at the SMPL spine1 joint height (umbilical-level proxy)"),
        "hip": ("torso", "max", np.linspace(y["pelvis"] - 0.4 * (y["pelvis"] - knee_y), y["pelvis"], steps),
                "maximum girth enclosing both buttocks, from the pelvis joint down toward the crotch"),
    }
    arms = [(J[L_SHOULDER], J[L_ELBOW]), (J[L_ELBOW], J[L_WRIST]),
            (J[R_SHOULDER], J[R_ELBOW]), (J[R_ELBOW], J[R_WRIST])]
    spine = [(J[PELVIS], J[NECK])]
    out = {}
    for name, (kind, pick, ts, definition) in specs.items():
        best = _search(v, faces, torso, ts, pick, exclude=arms)
        out[name] = _entry(best, definition, height, side=None)
    other_leg = {"left": [(J[R_HIP], J[R_KNEE]), (J[R_KNEE], J[R_ANKLE])],
                 "right": [(J[L_HIP], J[L_KNEE]), (J[L_KNEE], J[L_ANKLE])]}
    for side, (sh, el, wr, hp, kn, an) in {"left": (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE),
                                           "right": (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE)}.items():
        limbs = {
            "upper_arm": (lambda t, a=sh, b=el: limb(a, b, t), [0.5], "max",
                          "perpendicular to the upper arm at the shoulder-elbow midpoint"),
            "forearm": (lambda t, a=el, b=wr: limb(a, b, t), np.linspace(0.12, 0.45, 12), "max",
                        "maximum perpendicular girth in the proximal forearm (12-45% from elbow to wrist)"),
            "thigh": (lambda t, a=hp, b=kn: limb(a, b, t), [0.5], "max",
                      "perpendicular to the femur at the hip-knee joint midpoint (mid-thigh)"),
            "calf": (lambda t, a=kn, b=an: limb(a, b, t), np.linspace(0.15, 0.5, 12), "max",
                     "maximum perpendicular girth 15-50% from knee to ankle"),
        }
        for name, (fn, ts, pick, definition) in limbs.items():
            exclude = spine if name in ("upper_arm", "forearm") else other_leg[side]
            out[f"{name}_{side}"] = _entry(_search(v, faces, fn, ts, pick, exclude), definition, height, side=side)
    valid = sum(1 for m in out.values() if m["valid"])
    return {"units": "mesh units (relative; NOT calibrated centimetres)",
            "mesh_height": height,
            "method": "plane sections of the final mesh; tape = convex-hull perimeter of the enclosing loop",
            "valid_count": valid, "total": len(out), "measurements": out}


def _entry(best, definition, height, side):
    if best is None:
        return {"valid": False, "definition": definition, "reason": "no valid closed cross-section in the search range"}
    t, m = best
    return {"valid": True, "definition": definition, "search_parameter": float(t),
            "tape_circumference": m["tape_perimeter"], "contour_perimeter": m["perimeter"],
            "tape_circumference_over_height": m["tape_perimeter"] / height if height else None,
            "concavity_ratio": m["concavity_ratio"], "plane_origin": m["plane_origin"],
            "plane_normal": m["plane_normal"], "loops_in_plane": m["loops_in_plane"],
            "contour_points": int(m["n_points"]), "ring": m["points3d"]}


def write_rings_obj(path, measurements: dict, header="Measurement cross-sections (polylines)") -> int:
    """OBJ with one closed polyline ('l') per valid measurement, for overlay on the mesh."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {header}\n")
        offset = 1
        for name, m in measurements["measurements"].items():
            if not m.get("valid"):
                continue
            ring = np.asarray(m["ring"])
            fh.write(f"o {name}\n")
            for x, y, z in ring:
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            idx = list(range(offset, offset + len(ring))) + [offset]
            fh.write("l " + " ".join(map(str, idx)) + "\n")
            offset += len(ring)
            count += 1
    return count


def strip_rings(measurements: dict) -> dict:
    """JSON-friendly copy without the 3D ring point arrays."""
    clean = {k: v for k, v in measurements.items() if k != "measurements"}
    clean["measurements"] = {k: {kk: vv for kk, vv in m.items() if kk != "ring"}
                             for k, m in measurements["measurements"].items()}
    return clean


def circle_perimeter(r: float) -> float:
    return 2 * math.pi * r


# ------------------------------------------------------------ metric scale
MIN_HEIGHT_CM = 100.0
MAX_HEIGHT_CM = 250.0


def validate_height_cm(height_cm) -> float:
    """Operator-supplied barefoot standing height in centimetres."""
    if isinstance(height_cm, bool) or not isinstance(height_cm, (int, float)) or not math.isfinite(height_cm):
        raise ValueError("height_cm must be a finite number")
    if not MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM:
        raise ValueError(f"height_cm must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f}")
    return float(height_cm)


def scale_to_height(measurements: dict, height_cm, height_uncertainty_cm: float = 1.0) -> dict:
    """Convert relative circumferences (measurements.json) to centimetres using a known stature.

    Monocular reconstruction fixes shape but not absolute size, so one known length sets the scale:
    factor = height_cm / mesh_height. Every circumference scales by the same factor; a stature error of
    +-d cm therefore moves every value by the same relative amount (d / height). This is a scale
    transfer, not an accuracy guarantee for the shape itself.
    """
    height_cm = validate_height_cm(height_cm)
    mesh_height = measurements.get("mesh_height")
    if not isinstance(mesh_height, (int, float)) or not math.isfinite(mesh_height) or mesh_height <= 0:
        raise ValueError("measurements have no valid mesh_height to scale from")
    factor = height_cm / float(mesh_height)
    rel = float(height_uncertainty_cm) / height_cm
    out = {}
    for name, m in measurements.get("measurements", {}).items():
        if not m.get("valid") or not isinstance(m.get("tape_circumference"), (int, float)):
            out[name] = {"valid": False, "reason": m.get("reason", "invalid cross-section")}
            continue
        tape = m["tape_circumference"] * factor
        out[name] = {"valid": True, "tape_circumference_cm": round(tape, 1),
                     "contour_perimeter_cm": round(m["contour_perimeter"] * factor, 1)
                     if isinstance(m.get("contour_perimeter"), (int, float)) else None,
                     "height_uncertainty_contribution_cm": round(tape * rel, 2),
                     "definition": m.get("definition")}
    return {
        "units": "centimetres (scaled from operator-supplied height)",
        "height_cm": height_cm,
        "mesh_height_units": float(mesh_height),
        "cm_per_mesh_unit": factor,
        "height_uncertainty_cm_assumed": float(height_uncertainty_cm),
        "relative_scale_uncertainty": rel,
        "valid_count": sum(1 for m in out.values() if m["valid"]),
        "total": len(out),
        "measurements": out,
        "method": ("uniform scale so the final mesh's vertical extent equals the supplied barefoot height; "
                   "circumferences are the existing plane sections multiplied by the same factor"),
        "limitations": [
            "Absolute size comes entirely from the supplied height; an error in it scales every value equally.",
            "Stature is the vertical extent of the measurement A-pose mesh (legs abducted 5 deg), which is "
            "a few millimetres shorter than upright standing stature.",
            "The SMPL surface has no hair or shoes, so the height must be barefoot; the fitted shape follows the "
            "clothed silhouette and can overstate girths under loose clothing.",
            "Not validated against tape measurements; not a Tanita BIA/BCM body-composition result.",
        ],
    }
