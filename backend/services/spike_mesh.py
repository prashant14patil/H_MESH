"""Minimal geometry validation and exclusive OBJ export for experiments.

Dimensions are in the supplied model's units; no metric calibration is implied.
This module loads no body-model assets.
"""
from pathlib import Path

import numpy as np


def validate_mesh(vertices, faces):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Vertices must have shape [N, 3] with at least three points")
    if not np.issubdtype(vertices.dtype, np.number) or not np.isfinite(vertices).all():
        raise ValueError("Vertices must be finite numeric values")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Faces must have shape [M, 3] with at least one triangle")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Face indices must be integers")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Face indices out of range")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
              | (faces[:, 2] == faces[:, 0])):
        raise ValueError("Faces must not contain repeated indices")
    triangles = vertices[faces]
    area_twice = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area_twice).all() or np.any(area_twice <= 1e-12):
        raise ValueError("Degenerate triangles found")
    bounds = vertices.max(axis=0) - vertices.min(axis=0)
    return {"vertices": int(len(vertices)), "faces": int(len(faces)),
            "bounding_dimensions_model_units": bounds.tolist(),
            "coordinate_units": "unknown; not calibrated"}


def export_obj(path, vertices, faces):
    """Validate then create a new OBJ; never overwrite an existing file."""
    info = validate_mesh(vertices, faces)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write("# Experimental mesh; model-space units, not calibrated\n")
        for x, y, z in vertices:
            output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            output.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return info
