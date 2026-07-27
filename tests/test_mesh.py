from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

_HAS_PROPERTY_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "trimesh", "vtkmodules")
)
@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
class MeshTests(unittest.TestCase):
    def test_body_geometry_fingerprint_has_a_cross_repository_golden_value(
        self,
    ) -> None:
        import numpy as np

        from sim_asset_tools.mesh import body_geometry_sha256

        mesh = SimpleNamespace(
            vertices=np.asarray(
                [
                    [0.7148085236549377, 0.0, 0.0],
                    [0.0, 0.727572500705719, 0.0],
                    [0.0, 0.0, 0.766837477684021],
                    [
                        -0.4121514856815338,
                        -0.6345754861831665,
                        -0.10000000149011612,
                    ],
                ],
                dtype=np.float32,
            ),
            faces=np.asarray(
                [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
                dtype=np.int32,
            ),
        )

        self.assertEqual(
            body_geometry_sha256([mesh]),
            "0aa8e3ec34b9e3784f29dd2103e3772a32f45ecf6eb7bc679679010864420d6b",
        )
        reordered = SimpleNamespace(
            vertices=mesh.vertices,
            faces=mesh.faces[::-1, ::-1],
        )
        self.assertNotEqual(
            body_geometry_sha256([reordered]),
            body_geometry_sha256([mesh]),
        )
        nonfinite = SimpleNamespace(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces,
        )
        nonfinite.vertices[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            body_geometry_sha256([nonfinite])

    def test_body_geometry_fingerprint_concatenates_meshes(self) -> None:
        import trimesh

        from sim_asset_tools.mesh import body_geometry_sha256

        left = trimesh.creation.box(extents=(0.5, 1.0, 1.0))
        right = trimesh.creation.box(extents=(0.5, 1.0, 1.0))
        left.apply_translation((-0.25, 0.0, 0.0))
        right.apply_translation((0.25, 0.0, 0.0))

        self.assertEqual(
            body_geometry_sha256([left, right]),
            body_geometry_sha256([trimesh.util.concatenate((left, right))]),
        )

    def test_validation_checks_orientation_of_every_component(self) -> None:
        import numpy as np

        from sim_asset_tools.mesh import validate_mesh

        class Mesh:
            is_watertight = True
            is_winding_consistent = True

            def __init__(self, *, reverse_second: bool) -> None:
                tetrahedron = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
                second_faces = faces[:, ::-1] if reverse_second else faces
                self.vertices = np.vstack((tetrahedron, tetrahedron + [2.0, 0.0, 0.0]))
                self.faces = np.vstack((faces, second_faces + 4))
                self.area_faces = np.ones(len(self.faces))

        self.assertEqual(validate_mesh(Mesh(reverse_second=False), watertight=True), [])
        self.assertEqual(
            validate_mesh(Mesh(reverse_second=True), watertight=True),
            ["mesh normals must face outward"],
        )


@unittest.skipUnless(_HAS_PROPERTY_DEPS, "requires mesh property dependencies")
class MeshPropertyTests(unittest.TestCase):
    def test_collision_and_obb_properties(self) -> None:
        import numpy as np
        import trimesh

        from sim_asset_tools.mesh import collision_properties, oriented_bounding_box

        box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        collision = collision_properties(box)
        obb = oriented_bounding_box(box)

        self.assertAlmostEqual(collision["volume"], 24.0)
        self.assertTrue(np.allclose(collision["center_of_mass"], [0, 0, 0]))
        self.assertTrue(
            np.allclose(
                collision["inertia_per_unit_mass"],
                np.diag([25.0 / 12.0, 20.0 / 12.0, 13.0 / 12.0]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(obb["axes"]) @ np.asarray(obb["axes"]).T,
                np.eye(3),
            )
        )
        self.assertTrue(np.allclose(sorted(obb["extents"]), [2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
