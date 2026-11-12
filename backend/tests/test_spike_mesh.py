"""Synthetic OBJ validation tests. No licensed model assets are required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.services.spike_mesh import export_obj, validate_mesh


class MeshTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)

    def test_export_valid_obj_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            result = export_obj(path, self.vertices, self.faces)
            self.assertEqual(result["vertices"], 3)
            self.assertEqual(result["faces"], 1)
            self.assertIn("f 1 2 3", path.read_text(encoding="utf-8"))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                export_obj(path, self.vertices, self.faces)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_meshes_fail(self):
        bad_vertices = self.vertices.copy()
        bad_vertices[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_mesh(bad_vertices, self.faces)
        with self.assertRaises(ValueError):
            validate_mesh(self.vertices, np.array([[0, 1, 3]]))
        with self.assertRaises(ValueError):
            validate_mesh(self.vertices, np.array([[0, 1, 1]]))
        with self.assertRaises(ValueError):
            validate_mesh(self.vertices, np.array([[0., 1., 2.]]))
        with self.assertRaises(ValueError):
            validate_mesh(np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), self.faces)


if __name__ == "__main__":
    unittest.main()
