"""Model-free torso girths from the multi-view silhouettes (visual hull), compared with the fitted mesh.

For a completed run, every selected view has a fitted camera and body orientation. A horizontal slice of the
torso in the body's own (rest-pose) frame is carried into each view with that view's fitted spine/pelvis
transform and projected; a slice point belongs to the visual hull when it lies inside the silhouette in every
view (``--tolerance`` views may disagree, for mask noise). The convex perimeter of the hull slice is what a tape
would measure around the CLOTHED body - an upper bound on the true girth that does not depend on SMPL's shape
space, the shape prior or any learned regressor. A fitted body girth above it is impossible; the gap below it is
clothing (ease, drape over concavities) plus any body-model error.

    .venv\\Scripts\\python.exe scripts/visual_hull_girths.py RUN_ID [--height-cm 179] [--tolerance 1] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measurement_accuracy_study as mas  # noqa: E402  (SMPL loader shared with the accuracy study)
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

SPINE_CHAIN = [0, 3, 6, 9, 12]  # pelvis, spine1, spine2, spine3, neck (SMPL joint ids)


def load(run_id: str):
    d = next((ROOT / "outputs" / "reconstructions").glob(run_id + "*"))
    z = np.load(d / "diagnostics" / "fit_inputs.npz")
    f = np.load(d / "diagnostics" / "fit_result.npz")
    shape = tuple(int(x) for x in z["mask_shape"])
    masks = [np.unpackbits(z["masks_packed"][i])[:shape[0] * shape[1]].reshape(shape).astype(bool)
             if z["mask_usable"][i] and f["sil_weight"][i] > 0 else None for i in range(len(z["selected"]))]
    return d, z, f, masks


def joint_transforms(smpl, betas, global_orient, body_pose, device):
    """Rest-pose vertices/joints for ``betas`` and, per view, the 4x4 rest->posed transform of every joint."""
    from smplx.lbs import batch_rigid_transform
    f32 = dict(dtype=torch.float32, device=device)
    b = torch.as_tensor(betas, **f32)[None]
    V = len(global_orient)
    with torch.no_grad():
        rest = smpl(betas=b, global_orient=torch.eye(3, **f32)[None, None],
                    body_pose=torch.eye(3, **f32).expand(1, 23, 3, 3), pose2rot=False)
        v_rest = rest.vertices[0]
        j_rest = smpl.J_regressor.to(device) @ v_rest
        rots = torch.cat([torch.as_tensor(global_orient, **f32)[:, None], torch.as_tensor(body_pose, **f32)], 1)
        _, A = batch_rigid_transform(rots, j_rest[None].expand(V, -1, -1).contiguous(), smpl.parents.to(device))
        # Check: rigid-skinning the rest mesh with these transforms reproduces the posed mesh (pose blend
        # shapes aside), so slice points are carried into each view exactly like the fitted body is.
        W = smpl.lbs_weights.to(device)
        T = torch.einsum("vj,njab->nvab", W, A)
        vh = torch.cat([v_rest, torch.ones_like(v_rest[:, :1])], 1)
        skinned = torch.einsum("nvab,vb->nva", T, vh)[..., :3]
        posed = smv.posed_output(smpl, betas, global_orient, body_pose, device).vertices
        err = float((skinned - posed).norm(dim=-1).median())
    return v_rest.cpu().numpy().astype(np.float64), j_rest.cpu().numpy().astype(np.float64), \
        A.cpu().numpy().astype(np.float64), err


ARM_JOINTS = (16, 17, 18, 19, 20, 21, 22, 23)  # shoulders (upper arm), elbows, wrists, hands


def carve_arms(smpl, f, z, masks, margin_frac, device):
    """Remove each view's projected ARM region from its silhouette, except where it overlaps the fitted torso
    grown by ``margin_frac`` x box height (so clothing just outside the fitted torso is kept). Without this the
    hanging arms join the hull at chest level. The kept overlap uses the fitted body, so chest values from carved
    masks are an estimate, reported at two margins for sensitivity."""
    faces = np.asarray(smpl.faces, dtype=np.int64)
    part = smpl.lbs_weights.argmax(1).cpu().numpy()
    face_part = part[faces]
    arm_f = faces[np.isin(face_part, ARM_JOINTS).all(1)]
    torso_f = faces[~np.isin(face_part, ARM_JOINTS).any(1)]
    verts = smv.posed_output(smpl, f["betas"], f["global_orient"], f["body_pose"], device).vertices.cpu().numpy()
    out = []
    for i, m in enumerate(masks):
        if m is None:
            out.append(None)
            continue
        uv = smv.project_np(verts[i].astype(np.float64) + f["cam_t"][i], float(z["focal"]), z["center"][i])
        arm = smv.rasterize_silhouette(uv, arm_f, m.shape).astype(bool)
        torso = smv.rasterize_silhouette(uv, torso_f, m.shape)
        r = max(1, int(round(margin_frac * float(z["box_height"][i]))))
        torso = cv2.dilate(torso, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
        arm = cv2.dilate(arm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        out.append(m & ~(arm & ~torso))
    return out


def hull_slice(y, j_rest, A, cam_t, focal, centers, masks, tolerance, step=0.0025, half=0.40):
    """Occupancy grid of the visual hull in the horizontal rest-frame plane at height ``y``."""
    below = [j for j in SPINE_CHAIN if j_rest[j, 1] <= y]
    joint = below[-1] if below else SPINE_CHAIN[0]
    cx, cz = j_rest[joint, 0], j_rest[joint, 2]
    xs = np.arange(cx - half, cx + half, step)
    zs = np.arange(cz - half, cz + half, step)
    gx, gz = np.meshgrid(xs, zs, indexing="xy")
    pts = np.stack([gx.ravel(), np.full(gx.size, y), gz.ravel(), np.ones(gx.size)], 1)
    votes = np.zeros(len(pts), int)
    n = 0
    for i, m in enumerate(masks):
        if m is None:
            continue
        n += 1
        pc = (pts @ A[i, joint].T)[:, :3] + cam_t[i]
        uv = smv.project_np(pc, focal, centers[i])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
        inside = np.zeros(len(pts), bool)
        inside[ok] = m[v[ok], u[ok]]
        votes += inside
    occ = (votes >= n - tolerance).reshape(gx.shape)
    lab, k = _label(occ)
    ci, cj = int(np.argmin(np.abs(zs - cz))), int(np.argmin(np.abs(xs - cx)))
    comp = lab[ci, cj]
    if comp == 0:  # centre not occupied: take the largest component
        comp = int(np.argmax(np.bincount(lab.ravel())[1:]) + 1) if k else 0
    region = lab == comp if comp else np.zeros_like(occ)
    return region, xs, zs, joint, n


def _label(mask):
    k, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return lab, k - 1


def convex_perimeter(region, xs, zs):
    if region.sum() < 3:
        return None, None
    ii, jj = np.nonzero(region)
    step = xs[1] - xs[0]
    p = np.stack([xs[jj], zs[ii]], 1)
    # cell corners so the perimeter is not biased low by half a cell
    p = np.concatenate([p + [dx, dz] for dx in (-step / 2, step / 2) for dz in (-step / 2, step / 2)])
    hull = bm._hull(p)
    return bm._perimeter(hull), hull


def mesh_section(v_rest, faces, j_rest, joint, y):
    origin = np.array([j_rest[joint, 0], y, j_rest[joint, 2]])
    s = bm.measure_section(v_rest, faces, origin, np.array([0.0, 1.0, 0.0]))
    if not s.get("valid") and "tape_perimeter" not in s:
        return None, None
    pts = np.asarray(s["points3d"])[:, [0, 2]]
    return s["tape_perimeter"], pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--tolerance", type=int, default=1, help="views allowed to disagree (mask noise)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--carve-arms", type=float, default=None,
                    help="remove projected arms from the masks; value = torso margin as a fraction of box height")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = mas.load_smpl(dev)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    d, z, f, masks = load(a.run_id)
    v_rest, j_rest, A, err = joint_transforms(smpl, f["betas"], f["global_orient"], f["body_pose"], dev)
    va, _, _, _ = smv.canonical_mesh(smpl, f["betas"], np.tile(np.eye(3), (1, 23, 1, 1)), None, dev, pose="measurement")
    H = float(va[:, 1].max() - va[:, 1].min())
    tag = f"tol{a.tolerance}"
    if a.carve_arms is not None:
        masks = carve_arms(smpl, f, z, masks, a.carve_arms, dev)
        tag += f"_arms{a.carve_arms:g}"
    out = Path(a.out) if a.out else d / "diagnostics" / "visual_hull"
    out.mkdir(parents=True, exist_ok=True)

    y0, y1 = j_rest[0, 1] - 0.14, j_rest[9, 1] + 0.10
    rows, tiles = [], []
    for y in np.arange(y0, y1, 0.01):
        region, xs, zs, joint, n = hull_slice(y, j_rest, A, f["cam_t"], float(z["focal"]), z["center"], masks,
                                              a.tolerance)
        hp, hull = convex_perimeter(region, xs, zs)
        mp, mpts = mesh_section(v_rest, faces, j_rest, joint, y)
        width = float(np.ptp(hull[:, 0])) if hull is not None else None
        depth = float(np.ptp(hull[:, 1])) if hull is not None else None
        rows.append({"y_rel_pelvis_over_H": float((y - j_rest[0, 1]) / H), "joint": joint, "views": n,
                     "hull_girth_over_H": hp / H if hp else None, "mesh_girth_over_H": mp / H if mp else None,
                     "mesh_over_hull": (mp / hp) if (mp and hp) else None,
                     "hull_width_over_H": width / H if width else None, "hull_depth_over_H": depth / H if depth else None})
        if hull is not None and abs(round((y - y0) / 0.01)) % 3 == 0:
            tiles.append(_draw(region, xs, zs, hull, mpts, f"y-pelvis {(y - j_rest[0, 1]) / H:+.3f}H"))
    if tiles:
        cols = 6
        tiles += [np.full_like(tiles[0], 255)] * ((-len(tiles)) % cols)
        cv2.imwrite(str(out / f"hull_slices_{tag}.png"),
                    np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))
    res = {"run_id": d.name, "views_used": int(sum(m is not None for m in masks)), "tolerance": a.tolerance,
           "skinning_check_median_error": err, "mesh_height_smpl_units": H, "rows": rows,
           "note": "hull = convex perimeter of the multi-view visual hull of the CLOTHED body (upper bound)"}
    (out / f"visual_hull_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    hc = a.height_cm
    print(f"run {d.name[:8]}  views {res['views_used']}  tolerance {a.tolerance}  skinning check {err * 1000:.2f} mm")
    print(" y-pelvis/H joint  hull/H  mesh/H  mesh/hull   hull W/H  hull D/H" + ("   hull cm  mesh cm" if hc else ""))
    for r in rows:
        if r["hull_girth_over_H"] is None:
            continue
        line = (f"  {r['y_rel_pelvis_over_H']:+.3f}   {r['joint']:2d}   {r['hull_girth_over_H']:.3f}  "
                f"{(r['mesh_girth_over_H'] or float('nan')):.3f}   {(r['mesh_over_hull'] or float('nan')):.3f}     "
                f"{r['hull_width_over_H']:.3f}    {r['hull_depth_over_H']:.3f}")
        if hc:
            line += f"   {r['hull_girth_over_H'] * hc:6.1f}  {(r['mesh_girth_over_H'] or float('nan')) * hc:6.1f}"
        print(line)


def _draw(region, xs, zs, hull, mesh_pts, label, size=220):
    img = np.full((size, size, 3), 255, np.uint8)
    sx = size / (xs[-1] - xs[0])

    def px(p):
        return np.stack([(p[:, 0] - xs[0]) * sx, size - (p[:, 1] - zs[0]) * sx], 1).astype(np.int32)
    ii, jj = np.nonzero(region)
    img[np.clip(size - 1 - (ii * size // len(zs)), 0, size - 1), np.clip(jj * size // len(xs), 0, size - 1)] = (200, 200, 255)
    cv2.polylines(img, [px(hull)], True, (0, 0, 220), 1)
    if mesh_pts is not None:
        cv2.polylines(img, [px(mesh_pts)], True, (0, 150, 0), 1)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


if __name__ == "__main__":
    main()
