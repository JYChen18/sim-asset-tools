from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HAS_MESH_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("coacd", "numpy", "trimesh", "vtkmodules")
)


@unittest.skipUnless(_HAS_MESH_DEPS, "requires mesh dependencies")
class ObjectWorkflowTests(unittest.TestCase):
    def test_explicit_mjcf_inertial_matches_density_inference(self) -> None:
        import mujoco
        import numpy as np
        import trimesh
        from xml.etree import ElementTree as ET

        from sim_asset_tools.formats.mjcf import write_object_mjcf
        from sim_asset_tools.mesh import collision_properties

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            meshes = [
                trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
                trimesh.creation.box(extents=[0.3, 0.8, 0.5]),
            ]
            meshes[0].apply_translation([0.2, -0.3, 0.1])
            meshes[1].apply_translation([1.1, 0.2, -0.4])
            paths = []
            for index, mesh in enumerate(meshes):
                path = root / f"part_{index}.obj"
                mesh.export(path)
                paths.append(path)

            inferred_root = ET.Element("mujoco")
            assets = ET.SubElement(inferred_root, "asset")
            worldbody = ET.SubElement(inferred_root, "worldbody")
            body = ET.SubElement(worldbody, "body", {"name": "object"})
            for index, path in enumerate(paths):
                name = f"part_{index}"
                ET.SubElement(
                    assets,
                    "mesh",
                    {"name": name, "file": path.as_posix()},
                )
                ET.SubElement(
                    body,
                    "geom",
                    {"type": "mesh", "mesh": name},
                )
            inferred = mujoco.MjModel.from_xml_string(
                ET.tostring(inferred_root, encoding="unicode")
            )

            explicit_path = root / "explicit.xml"
            write_object_mjcf(
                explicit_path,
                paths[0],
                paths,
                collision_properties(trimesh.util.concatenate(meshes)),
            )
            explicit = mujoco.MjSpec.from_file(explicit_path.as_posix()).compile()
            inferred_id = mujoco.mj_name2id(
                inferred,
                mujoco.mjtObj.mjOBJ_BODY,
                "object",
            )
            explicit_id = mujoco.mj_name2id(
                explicit,
                mujoco.mjtObj.mjOBJ_BODY,
                "object",
            )

            self.assertTrue(
                np.isclose(
                    inferred.body_mass[inferred_id],
                    explicit.body_mass[explicit_id],
                    rtol=1.0e-6,
                    atol=1.0e-5,
                )
            )
            self.assertTrue(
                np.allclose(
                    inferred.body_ipos[inferred_id],
                    explicit.body_ipos[explicit_id],
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            )
            self.assertTrue(
                np.allclose(
                    inferred.body_inertia[inferred_id],
                    explicit.body_inertia[explicit_id],
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            )

            def inertia_tensor(model, body_id):
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
                rotation = rotation.reshape((3, 3))
                return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T

            self.assertTrue(
                np.allclose(
                    inertia_tensor(inferred, inferred_id),
                    inertia_tensor(explicit, explicit_id),
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            )

    def test_overwrite_rejects_an_output_that_contains_the_input(self) -> None:
        from sim_asset_tools.workflows import prepare_object

        with tempfile.TemporaryDirectory() as value:
            output = Path(value)
            source = output / "mesh.obj"
            source.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not contain"):
                prepare_object(source, output, overwrite=True)

            self.assertTrue(source.is_file())

    def test_batch_rejects_inputs_that_share_an_output_directory(self) -> None:
        from sim_asset_tools.workflows import prepare_objects

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "cup.obj").touch()
            (root / "cup.stl").touch()

            with self.assertRaisesRegex(ValueError, "share output directories"):
                prepare_objects(root, root / "output")

    def test_prepare_object_fingerprints_each_single_model_format(self) -> None:
        import trimesh

        from sim_asset_tools.mesh import load_mesh
        from sim_asset_tools.workflows import check_object, prepare_object
        from sim_asset_tools.workflows import object as object_workflow

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "box.obj"
            trimesh.creation.box().export(source)

            def copy_source(input_path, output_path, *_args):
                load_mesh(input_path).export(output_path)
                return output_path

            def keep_one_collision(mesh, **_kwargs):
                return [mesh.copy()]

            for model_format, model_name in (
                ("mjcf", "model.xml"),
                ("urdf", "model.urdf"),
            ):
                with (
                    self.subTest(model_format=model_format),
                    mock.patch.object(
                        object_workflow,
                        "prepare_surface",
                        side_effect=copy_source,
                    ),
                    mock.patch.object(
                        object_workflow,
                        "decompose_mesh",
                        side_effect=keep_one_collision,
                    ),
                ):
                    result = prepare_object(
                        source,
                        root / model_format,
                        formats=(model_format,),
                    )
                    manifest = json.loads(
                        result.manifest_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(manifest["models"], {model_name: ["object"]})
                    self.assertEqual(check_object(result.output_directory), [])

    def test_prepare_object_writes_a_valid_self_describing_bundle(self) -> None:
        import trimesh
        from xml.etree import ElementTree as ET

        from sim_asset_tools.formats.manifest import sha256_file, sha256_json
        from sim_asset_tools.formats.object_manifest import OBJECT_MANIFEST_SCHEMA
        from sim_asset_tools.mesh import load_mesh
        from sim_asset_tools.workflows import check_object, prepare_object
        from sim_asset_tools.workflows import object as object_workflow

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "box.obj"
            trimesh.creation.box().export(source)

            def copy_source(input_path, output_path, *_args):
                load_mesh(input_path).export(output_path)
                return output_path

            with mock.patch.object(
                object_workflow, "prepare_surface", side_effect=copy_source
            ):
                result = prepare_object(source, root / "asset")

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            pristine_manifest = copy.deepcopy(manifest)
            collision_contents = {
                path.name: path.read_bytes()
                for path in (result.output_directory / "collision").glob("*.obj")
            }
            self.assertEqual(manifest["schema"], OBJECT_MANIFEST_SCHEMA)
            self.assertEqual(
                set(manifest),
                {"schema", "geometry", "models", "recipe", "sha256", "surfaces"},
            )
            self.assertEqual(
                manifest["models"],
                {"model.urdf": ["object"], "model.xml": ["object"]},
            )
            self.assertEqual(manifest["surfaces"], {"object": "visual.obj"})
            geometry = manifest["geometry"]
            self.assertEqual(geometry["source_aabb_extents"], [1.0, 1.0, 1.0])
            self.assertEqual(len(geometry["obb_center"]), 3)
            self.assertEqual(len(geometry["obb_axes"]), 3)
            self.assertEqual(len(geometry["obb_extents"]), 3)
            self.assertEqual(
                set(geometry),
                {
                    "source_aabb_center",
                    "source_aabb_extents",
                    "obb_center",
                    "obb_axes",
                    "obb_extents",
                },
            )

            hashes = manifest["sha256"]
            self.assertEqual(
                set(hashes),
                {"bodies", "files", "metadata", "self"},
            )
            self.assertEqual(set(hashes["bodies"]), {"object"})
            self.assertEqual(
                set(hashes["metadata"]),
                {"schema", "geometry", "models", "recipe", "surfaces"},
            )
            self.assertEqual(
                hashes["self"],
                sha256_json(
                    {name: hashes[name] for name in ("files", "bodies", "metadata")}
                ),
            )
            self.assertIn("source.obj", hashes["files"])
            self.assertIn("visual.obj", hashes["files"])
            self.assertIn("model.xml", hashes["files"])
            self.assertIn("model.urdf", hashes["files"])
            self.assertFalse(
                any(path.startswith("collision/") for path in hashes["files"])
            )

            self.assertTrue(result.mjcf_path.is_file())
            self.assertTrue(result.urdf_path.is_file())
            self.assertEqual(result.mjcf_path.parent, result.output_directory)
            self.assertEqual(result.urdf_path.parent, result.output_directory)
            self.assertEqual(check_object(result.output_directory), [])

            model_root = ET.parse(result.mjcf_path).getroot()
            inertial = model_root.find("./worldbody/body/inertial")
            self.assertIsNotNone(inertial)
            self.assertIsNotNone(inertial.get("mass"))
            self.assertEqual(len(inertial.get("pos").split()), 3)
            self.assertEqual(len(inertial.get("fullinertia").split()), 6)
            self.assertTrue(
                all(
                    geom.get("density") == "0"
                    for geom in model_root.findall("./worldbody/body/geom")
                    if geom.get("name", "").startswith("collision_")
                )
            )

            collision_path = next((result.output_directory / "collision").glob("*.obj"))
            original_collision = collision_path.read_bytes()
            collision_path.write_bytes(original_collision + b"\n")
            self.assertEqual(check_object(result.output_directory), [])

            collision_mesh = load_mesh(collision_path)
            collision_mesh.apply_scale(1.2)
            collision_mesh.export(collision_path)
            collision_errors = check_object(result.output_directory)
            self.assertTrue(
                any(
                    "body object fingerprint does not match" in error
                    for error in collision_errors
                ),
                collision_errors,
            )
            collision_path.write_bytes(original_collision)
            self.assertEqual(check_object(result.output_directory), [])

            manifest["surfaces"]["object"] = "source.obj"
            result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(
                any(
                    "metadata surfaces fingerprint does not match" in error
                    for error in check_object(result.output_directory)
                )
            )

            manifest["surfaces"]["object"] = "visual.obj"
            manifest["geometry"]["source_aabb_extents"][0] *= 2.0
            result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(
                any(
                    "metadata geometry fingerprint does not match" in error
                    for error in check_object(result.output_directory)
                )
            )

            def publish_rehashed(value: dict[str, object]) -> None:
                value_hashes = value["sha256"]
                assert isinstance(value_hashes, dict)
                files = value_hashes["files"]
                metadata = value_hashes["metadata"]
                assert isinstance(files, dict)
                assert isinstance(metadata, dict)
                for relative in tuple(files):
                    path = result.output_directory / relative
                    if path.is_file():
                        files[relative] = sha256_file(path)
                for name in ("schema", "geometry", "models", "recipe", "surfaces"):
                    metadata[name] = sha256_json(value[name])
                value_hashes["self"] = sha256_json(
                    {
                        name: value_hashes[name]
                        for name in ("files", "bodies", "metadata")
                    }
                )
                result.manifest_path.write_text(
                    json.dumps(value),
                    encoding="utf-8",
                )

            manifest = copy.deepcopy(pristine_manifest)
            manifest["geometry"] = []
            publish_rehashed(manifest)
            self.assertIn(
                "manifest geometry must be an object",
                check_object(result.output_directory),
            )

            manifest = copy.deepcopy(pristine_manifest)
            visual_path = result.output_directory / "visual.obj"
            visual_bytes = visual_path.read_bytes()
            visual_path.write_text("not an OBJ mesh\n", encoding="utf-8")
            publish_rehashed(manifest)
            self.assertTrue(
                any(
                    error.startswith("visual could not be loaded:")
                    for error in check_object(result.output_directory)
                )
            )

            visual_path.write_bytes(visual_bytes)
            manifest = copy.deepcopy(pristine_manifest)
            for path in (result.output_directory / "collision").glob("*.obj"):
                path.unlink()
            publish_rehashed(manifest)
            self.assertIn(
                "collision directory must contain at least one OBJ mesh",
                check_object(result.output_directory),
            )

            for name, content in collision_contents.items():
                (result.output_directory / "collision" / name).write_bytes(content)
            (result.output_directory / "collision" / "notes.txt").write_text(
                "unexpected\n",
                encoding="utf-8",
            )
            manifest = copy.deepcopy(pristine_manifest)
            publish_rehashed(manifest)
            self.assertIn(
                "unexpected collision artifact: collision/notes.txt",
                check_object(result.output_directory),
            )

            (result.output_directory / "collision" / "notes.txt").unlink()
            result.manifest_path.write_text(
                json.dumps(pristine_manifest),
                encoding="utf-8",
            )
            collision_link = result.output_directory / "collision" / "outside.obj"
            try:
                collision_link.symlink_to(source)
            except (NotImplementedError, OSError):
                pass
            else:
                errors = check_object(result.output_directory)
                self.assertTrue(
                    any("symbolic link" in error for error in errors),
                    errors,
                )
                collision_link.unlink()

            result.manifest_path.write_text(
                json.dumps(pristine_manifest),
                encoding="utf-8",
            )
            model_root = ET.parse(result.mjcf_path).getroot()
            model_root.find("./worldbody/body/inertial").set("mass", "1")
            ET.ElementTree(model_root).write(result.mjcf_path, encoding="unicode")
            manifest = copy.deepcopy(pristine_manifest)
            publish_rehashed(manifest)
            self.assertTrue(
                any(
                    "inertial mass does not match collision geometry" in error
                    for error in check_object(result.output_directory)
                )
            )

    def test_check_object_rejects_inconsistent_models(self) -> None:
        import trimesh
        from xml.etree import ElementTree as ET

        from sim_asset_tools.formats.manifest import sha256_file, sha256_json
        from sim_asset_tools.mesh import load_mesh
        from sim_asset_tools.workflows import check_object, prepare_object
        from sim_asset_tools.workflows import object as object_workflow

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "box.obj"
            trimesh.creation.box().export(source)

            def copy_source(input_path, output_path, *_args):
                load_mesh(input_path).export(output_path)
                return output_path

            def keep_one_collision(mesh, **_kwargs):
                return [mesh.copy()]

            with (
                mock.patch.object(
                    object_workflow,
                    "prepare_surface",
                    side_effect=copy_source,
                ),
                mock.patch.object(
                    object_workflow,
                    "decompose_mesh",
                    side_effect=keep_one_collision,
                ),
            ):
                result = prepare_object(source, root / "asset")

            pristine_manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            pristine_mjcf = result.mjcf_path.read_bytes()
            pristine_urdf = result.urdf_path.read_bytes()

            def publish_model_change(model_path: Path) -> None:
                manifest = copy.deepcopy(pristine_manifest)
                files = manifest["sha256"]["files"]
                files[model_path.name] = sha256_file(model_path)
                hashes = manifest["sha256"]
                hashes["self"] = sha256_json(
                    {name: hashes[name] for name in ("files", "bodies", "metadata")}
                )
                result.manifest_path.write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )

            mjcf = ET.parse(result.mjcf_path)
            ET.SubElement(
                mjcf.getroot().find("./worldbody"), "body", {"name": "hidden"}
            )
            mjcf.write(result.mjcf_path, encoding="unicode")
            publish_model_change(result.mjcf_path)
            self.assertTrue(
                any(
                    "only one body named 'object'" in error
                    for error in check_object(result.output_directory)
                )
            )

            result.mjcf_path.write_bytes(pristine_mjcf)
            mjcf = ET.parse(result.mjcf_path)
            mjcf.getroot().find("./asset/mesh").set("file", source.as_posix())
            mjcf.write(result.mjcf_path, encoding="unicode")
            publish_model_change(result.mjcf_path)
            self.assertTrue(
                any(
                    "must be portable and relative" in error
                    for error in check_object(result.output_directory)
                )
            )

            result.mjcf_path.write_bytes(pristine_mjcf)
            mjcf = ET.parse(result.mjcf_path)
            mjcf.getroot().find("./worldbody/body").set("pos", "1 2 3")
            mjcf.write(result.mjcf_path, encoding="unicode")
            publish_model_change(result.mjcf_path)
            self.assertTrue(
                any(
                    "MJCF object body must use an identity pose" in error
                    for error in check_object(result.output_directory)
                )
            )

            result.mjcf_path.write_bytes(pristine_mjcf)
            result.urdf_path.write_bytes(pristine_urdf)
            urdf = ET.parse(result.urdf_path)
            link = urdf.getroot().find("link")
            link.append(copy.deepcopy(link.find("collision")))
            urdf.write(result.urdf_path, encoding="unicode")
            publish_model_change(result.urdf_path)
            self.assertTrue(
                any(
                    "model collision geometry does not match" in error
                    for error in check_object(result.output_directory)
                )
            )

            result.urdf_path.write_bytes(pristine_urdf)
            urdf = ET.parse(result.urdf_path)
            visual = urdf.getroot().find("./link/visual")
            visual.append(copy.deepcopy(visual.find("geometry")))
            urdf.write(result.urdf_path, encoding="unicode")
            publish_model_change(result.urdf_path)
            self.assertTrue(
                any(
                    "must contain exactly one geometry and one mesh" in error
                    for error in check_object(result.output_directory)
                )
            )

    def test_check_object_reports_malformed_records(self) -> None:
        from sim_asset_tools import cli
        from sim_asset_tools.formats.object_manifest import OBJECT_MANIFEST_SCHEMA
        from sim_asset_tools.workflows import check_object

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "asset.json").write_text(
                json.dumps(
                    {
                        "schema": OBJECT_MANIFEST_SCHEMA,
                        "geometry": "invalid",
                        "models": "invalid",
                        "sha256": {},
                        "recipe": {},
                        "surfaces": "invalid",
                    }
                ),
                encoding="utf-8",
            )

            errors = check_object(root)

            self.assertIn(
                "manifest sha256 is missing the metadata geometry fingerprint",
                errors,
            )
            self.assertIn("manifest sha256 is missing the self fingerprint", errors)
            self.assertIn("manifest geometry must be an object", errors)
            self.assertEqual(cli.main(["check", "object", str(root)]), 1)

    def test_check_object_rejects_unknown_schema(self) -> None:
        from sim_asset_tools.workflows import check_object

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "asset.json").write_text(
                json.dumps({"schema": "sim-asset/object/v2"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Regenerate the object bundle"):
                check_object(root)


if __name__ == "__main__":
    unittest.main()
