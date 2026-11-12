"""Model-independent mesh diagnostics and point-cloud alignment."""
import hashlib
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from backend.services.spike_mesh import validate_mesh


def load_mesh(path):
    mesh = trimesh.load_mesh(str(Path(path)), file_type="obj", process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("OBJ is not a single triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    validate_mesh(vertices, faces)
    return mesh, vertices, faces


def diagnostics(mesh, vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = float(np.linalg.norm(cross, axis=1).sum() / 2)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    components = mesh.split(only_watertight=False)
    volume_valid = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume)
    return {
        "vertices": len(vertices), "faces": len(faces),
        "centroid_vertex_mean": vertices.mean(axis=0).tolist(),
        "bbox_min": vertices.min(axis=0).tolist(), "bbox_max": vertices.max(axis=0).tolist(),
        "bbox_dimensions": np.ptp(vertices, axis=0).tolist(), "surface_area": area,
        "signed_volume": float(mesh.volume) if volume_valid else None,
        "volume_note": "closed, consistently wound positive volume only; self-intersections not checked" if volume_valid else "unavailable: not a closed consistently wound positive-volume mesh",
        "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
        "components": len(components), "component_face_counts": sorted([len(c.faces) for c in components], reverse=True),
        "boundary_edges": int(np.sum(counts == 1)), "nonmanifold_edges": int(np.sum(counts > 2)),
        "unused_vertices": int(len(vertices) - len(np.unique(faces))),
        "ordered_face_sha256": hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest(),
        "units": "model units; not calibrated",
    }


def symmetric_distances(source, target):
    """Symmetric nearest-vertex distances (not surface distances)."""
    d1 = cKDTree(target).query(source)[0]
    d2 = cKDTree(source).query(target)[0]
    return {"mean": float((d1.mean() + d2.mean()) / 2),
            "rms": float(np.sqrt((np.mean(d1*d1) + np.mean(d2*d2)) / 2)),
            "p95": float(np.percentile(np.concatenate((d1, d2)), 95))}


def fit_corresponding(source, target, similarity=False):
    """Least-squares proper rigid/similarity fit for explicit point correspondences."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Expected matching Nx3 point arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Points must be finite")
    a, b = source.mean(axis=0), target.mean(axis=0)
    x, y = source - a, target - b
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError("Degenerate correspondences")
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(x*x)) if similarity else 1.0
    if scale <= 0:
        raise ValueError("Invalid similarity scale")
    translation = b - scale * a @ rotation
    return scale, rotation, translation


def apply(points, transform):
    scale, rotation, translation = transform
    return scale * points @ rotation + translation


def icp(source, target, *, similarity=False, iterations=50, sample_size=2500):
    """Deterministic multistart point-to-point ICP; nearest vertices, no index correspondence."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    src = source[np.linspace(0, len(source)-1, min(len(source), sample_size), dtype=int)]
    dst = target[np.linspace(0, len(target)-1, min(len(target), sample_size), dtype=int)]
    tree = cKDTree(dst)
    best = None
    # Four initial rotations around Y; reference coordinate convention is unknown.
    for theta in (0, np.pi/2, np.pi, 3*np.pi/2):
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)
        scale = 1.0
        trans = dst.mean(axis=0) - src.mean(axis=0) @ rot
        previous = float("inf")
        used = 0
        for used in range(1, iterations + 1):
            moved = scale * src @ rot + trans
            nearest, indexes = tree.query(moved)
            # Trim 10% of largest residuals to reduce occlusion/outlier influence.
            keep = nearest <= np.quantile(nearest, .9)
            try:
                s_delta, r_delta, t_delta = fit_corresponding(moved[keep], dst[indexes[keep]], similarity)
            except ValueError:
                break
            scale = s_delta * scale
            rot = rot @ r_delta
            trans = s_delta * trans @ r_delta + t_delta
            error = float(np.mean(nearest[keep]))
            if abs(previous - error) < 1e-7:
                break
            previous = error
        transformed = scale * source @ rot + trans
        metric = symmetric_distances(transformed, target)
        candidate = (metric["mean"], scale, rot, trans, used, theta, metric)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, scale, rot, trans, used, theta, metric = best
    return {"scale": float(scale), "rotation_row_vector": rot.tolist(),
            "translation": trans.tolist(), "iterations": used,
            "initial_yaw_deg": float(np.degrees(theta)), "distance": metric}


def centered(points):
    return points - points.mean(axis=0)


def bbox_normalized(source, target):
    """Translation plus uniform max-extent scale; does not alter proportions."""
    source = centered(source)
    target_c = centered(target)
    a, b = np.max(np.ptp(source, axis=0)), np.max(np.ptp(target_c, axis=0))
    if a <= 0 or b <= 0:
        raise ValueError("Degenerate bounding box")
    return source * (b/a) + target.mean(axis=0), float(b/a)
