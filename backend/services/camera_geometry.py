"""Model-independent pinhole cameras.

Convention: x_cam = R_world_to_camera @ x_world + t_world_to_camera;
+x right, +y down, +z forward. Pixels have origin at image top-left.
No lens distortion is modelled. Units are those of the supplied world points.
"""
from dataclasses import dataclass

import numpy as np


class CalibrationError(ValueError):
    """Invalid camera parameters or geometry."""


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    rotation: np.ndarray
    translation: np.ndarray
    provenance: str  # explicitly "synthetic", "approximate", or "calibrated"

    def __post_init__(self):
        try:
            values = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=float)
            rotation = np.array(self.rotation, dtype=float, copy=True)
            translation = np.array(self.translation, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("Camera values must be numeric") from exc
        if (values.shape != (4,) or not np.isfinite(values).all()
                or values[0] <= 0 or values[1] <= 0):
            raise CalibrationError("Focal lengths must be positive and intrinsics finite")
        if (isinstance(self.width, bool) or isinstance(self.height, bool)
                or not isinstance(self.width, (int, np.integer))
                or not isinstance(self.height, (int, np.integer))
                or self.width <= 0 or self.height <= 0):
            raise CalibrationError("Image dimensions must be positive integers")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise CalibrationError("Rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise CalibrationError("Rotation must be a proper orthonormal matrix")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise CalibrationError("Translation must be a finite 3-vector")
        if self.provenance not in {"synthetic", "approximate", "calibrated"}:
            raise CalibrationError("Explicit calibration provenance is required")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        for name, value in zip(("fx", "fy", "cx", "cy"), values):
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def center_world(self):
        return -self.rotation.T @ self.translation

    @property
    def projection_matrix(self):
        k = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        return k @ np.column_stack((self.rotation, self.translation))

    def world_to_camera(self, points):
        points = _points(points, 3)
        return points @ self.rotation.T + self.translation

    def camera_to_world(self, points):
        points = _points(points, 3)
        return (points - self.translation) @ self.rotation

    def project(self, points):
        camera_points = self.world_to_camera(points)
        if np.any(camera_points[..., 2] <= 1e-9):
            raise CalibrationError("Points must be in front of the camera")
        xy = camera_points[..., :2] / camera_points[..., 2:3]
        return xy * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])

    def rays(self, pixels):
        """Return world-space camera origin and unit directions for pixels."""
        pixels = _points(pixels, 2)
        xy = (pixels - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        direction = np.concatenate((xy, np.ones(xy.shape[:-1] + (1,))), axis=-1)
        direction = direction @ self.rotation
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        return self.center_world.copy(), direction


def _points(points, dimensions):
    try:
        values = np.asarray(points, dtype=float)
    except (ValueError, TypeError) as exc:
        raise CalibrationError("Points must be numeric") from exc
    if values.ndim < 1 or values.shape[-1] != dimensions or not np.isfinite(values).all():
        raise CalibrationError(f"Points must be finite with last dimension {dimensions}")
    return values


def transform_camera_points(points, source: Camera, destination: Camera):
    """Transform 3D points from source camera coordinates to destination camera coordinates."""
    return destination.world_to_camera(source.camera_to_world(points))


def reprojection_errors(camera: Camera, points, observations):
    """Euclidean pixel errors; callers must mask missing observations first."""
    projected = camera.project(points)
    observed = _points(observations, 2)
    if projected.shape != observed.shape:
        raise CalibrationError("Observation shape must match projected points")
    return np.linalg.norm(projected - observed, axis=-1)
