"""Camera geometry tests use synthetic points and no model assets."""
import unittest
import numpy as np

from backend.services.camera_geometry import Camera, CalibrationError, reprojection_errors, transform_camera_points


class CameraTests(unittest.TestCase):
    def camera(self, **changes):
        values = dict(fx=500., fy=600., cx=320., cy=240., width=640, height=480,
                      rotation=np.eye(3), translation=np.array([0., 0., 4.]), provenance="synthetic")
        values.update(changes)
        return Camera(**values)

    def test_projection_and_inverse_rays(self):
        cam = self.camera()
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        pixels = cam.project(points)
        np.testing.assert_allclose(pixels, [[320, 240], [420, 360]])
        origin, directions = cam.rays(pixels)
        relative = points - origin
        np.testing.assert_allclose(directions, relative / np.linalg.norm(relative, axis=1)[:, None])
        np.testing.assert_allclose(reprojection_errors(cam, points, pixels + [3, 4]), [5, 5])

    def test_coordinate_roundtrip_and_between_cameras(self):
        angle = 0.5
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        source = self.camera(rotation=rotation, translation=np.array([1., 2., 4.]))
        dest = self.camera(translation=np.array([-1., 0., 3.]))
        points = np.array([[.2, .3, .4], [1., 2., 3.]])
        np.testing.assert_allclose(source.camera_to_world(source.world_to_camera(points)), points)
        np.testing.assert_allclose(transform_camera_points(source.world_to_camera(points), source, dest),
                                   dest.world_to_camera(points))

    def test_invalid_calibration_and_behind_camera(self):
        for change in (dict(fx=0), dict(rotation=np.zeros((3, 3))), dict(rotation=-np.eye(3)),
                       dict(translation=[0, 0, float("nan")]), dict(provenance=""),
                       dict(width=0), dict(cx=float("nan"))):
            with self.subTest(change=change), self.assertRaises(CalibrationError):
                self.camera(**change)
        with self.assertRaises(CalibrationError):
            self.camera().project([[0, 0, -5]])


if __name__ == "__main__":
    unittest.main()
