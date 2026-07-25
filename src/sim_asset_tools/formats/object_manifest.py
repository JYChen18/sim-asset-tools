"""Read, write, and validate ``sim-asset/object`` manifests.

The object schema describes model-to-body ownership, one sampling surface per
body, source and oriented bounds, and the preparation recipe. Ordinary files
are hashed individually, while all collision pieces belonging to a body share
one canonical geometry fingerprint. Mass properties live in the generated
model rather than in this manifest.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree as ET

from .manifest import (
    MANIFEST_NAME,
    load_manifest,
    relative_artifact_path,
    resolve_artifact,
    sha256_file,
    sha256_json,
    verify_digest,
    verify_sha256_map,
    write_manifest,
)

OBJECT_MANIFEST_SCHEMA = "sim-asset/object"

_BODY_NAME = "object"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset(
    {"schema", "geometry", "models", "recipe", "sha256", "surfaces"}
)
_METADATA_KEYS = _TOP_LEVEL_KEYS - {"sha256"}
_SHA256_KEYS = frozenset({"files", "bodies", "metadata", "self"})
_GEOMETRY_KEYS = frozenset(
    {
        "source_aabb_center",
        "source_aabb_extents",
        "obb_center",
        "obb_axes",
        "obb_extents",
    }
)
_MJCF_POSE_ATTRIBUTES = frozenset(
    {"axisangle", "euler", "fromto", "pos", "quat", "xyaxes", "zaxis"}
)


def write_object_manifest(
    path: str | Path,
    *,
    source_aabb_center: list[float],
    source_aabb_extents: list[float],
    obb: dict[str, object],
    recipe: dict[str, object],
    surface: Path,
    models: Iterable[Path],
    artifacts: Iterable[Path],
    body_sha256: str,
) -> None:
    """Build, fingerprint, and atomically write an object manifest."""
    path = Path(path)
    root = path.parent
    geometry = {
        "source_aabb_center": source_aabb_center,
        "source_aabb_extents": source_aabb_extents,
        "obb_center": obb["center"],
        "obb_axes": obb["axes"],
        "obb_extents": obb["extents"],
    }
    surfaces = {_BODY_NAME: relative_artifact_path(root, surface)}
    model_records = {
        relative_artifact_path(root, model): [_BODY_NAME]
        for model in sorted(
            models,
            key=lambda model: relative_artifact_path(root, model),
        )
    }
    files = {
        relative_artifact_path(root, artifact): sha256_file(artifact)
        for artifact in sorted(
            artifacts,
            key=lambda artifact: relative_artifact_path(root, artifact),
        )
    }
    metadata_values = {
        "schema": OBJECT_MANIFEST_SCHEMA,
        "geometry": geometry,
        "models": model_records,
        "recipe": recipe,
        "surfaces": surfaces,
    }
    hashes: dict[str, object] = {
        "files": files,
        "bodies": {_BODY_NAME: body_sha256},
        "metadata": {
            name: sha256_json(value) for name, value in sorted(metadata_values.items())
        },
    }
    hashes["self"] = sha256_json(hashes)
    write_manifest(
        path,
        {
            **metadata_values,
            "sha256": hashes,
        },
    )


def check_object_manifest(path_or_directory: str | Path) -> list[str]:
    """Verify an object manifest, its artifacts, and its collision geometry."""
    manifest_path = Path(path_or_directory)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    schema = manifest.get("schema")
    if schema != OBJECT_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported object manifest schema {schema!r}; "
            f"expected {OBJECT_MANIFEST_SCHEMA!r}. Regenerate the object bundle."
        )

    errors: list[str] = []
    _check_metadata_shapes(manifest, errors)
    files, bodies = _check_hash_structure(manifest, errors)
    root = manifest_path.parent.resolve()
    errors.extend(verify_sha256_map(root, files))
    _check_manifest_relations(root, manifest, files, bodies, errors)
    _check_object_meshes(root, manifest, bodies, errors)
    return errors


def _check_metadata_shapes(
    manifest: dict[str, object],
    errors: list[str],
) -> None:
    if set(manifest) != _TOP_LEVEL_KEYS:
        errors.append(
            "object manifest must contain exactly: "
            + ", ".join(sorted(_TOP_LEVEL_KEYS))
        )

    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        errors.append("manifest geometry must be an object")
    else:
        _check_geometry_shape(geometry, errors)

    if not isinstance(manifest.get("models"), dict):
        errors.append("manifest models must be an object")
    if not isinstance(manifest.get("recipe"), dict):
        errors.append("manifest recipe must be an object")
    if not isinstance(manifest.get("surfaces"), dict):
        errors.append("manifest surfaces must be an object")


def _check_geometry_shape(
    geometry: dict[object, object],
    errors: list[str],
) -> None:
    if set(geometry) != _GEOMETRY_KEYS:
        errors.append(
            "manifest geometry must contain exactly: "
            + ", ".join(sorted(_GEOMETRY_KEYS))
        )
        return

    for name in ("source_aabb_center", "obb_center"):
        if not _is_numeric_array(geometry[name], (3,)):
            errors.append(f"manifest geometry {name} must be a finite 3-vector")

    source_extents = geometry["source_aabb_extents"]
    if not _is_numeric_array(source_extents, (3,)) or not all(
        item >= 0 for item in source_extents
    ):
        errors.append(
            "manifest geometry source_aabb_extents must be a nonnegative finite "
            "3-vector"
        )

    obb_extents = geometry["obb_extents"]
    if not _is_numeric_array(obb_extents, (3,)) or not all(
        item > 0 for item in obb_extents
    ):
        errors.append(
            "manifest geometry obb_extents must be a positive finite 3-vector"
        )
    if not _is_numeric_array(geometry["obb_axes"], (3, 3)):
        errors.append("manifest geometry obb_axes must be a finite 3-by-3 matrix")


def _check_hash_structure(
    manifest: dict[str, object],
    errors: list[str],
) -> tuple[dict[object, object], dict[object, object]]:
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        errors.append("manifest sha256 must be an object")
        return {}, {}
    if set(hashes) != _SHA256_KEYS:
        errors.append(
            "manifest sha256 must contain exactly: " + ", ".join(sorted(_SHA256_KEYS))
        )

    files = hashes.get("files")
    if not isinstance(files, dict):
        errors.append("manifest sha256 files must be an object")
        files = {}
    bodies = hashes.get("bodies")
    if not isinstance(bodies, dict):
        errors.append("manifest sha256 bodies must be an object")
        bodies = {}
    metadata = hashes.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("manifest sha256 metadata must be an object")
        metadata = {}

    if set(metadata) != _METADATA_KEYS:
        errors.append(
            "manifest sha256 metadata must contain exactly: "
            + ", ".join(sorted(_METADATA_KEYS))
        )
    for name in sorted(_METADATA_KEYS):
        try:
            actual = sha256_json(manifest.get(name))
        except (TypeError, ValueError):
            errors.append(f"manifest {name} cannot be fingerprinted as JSON")
            continue
        verify_digest(
            f"metadata {name}",
            metadata.get(name),
            actual,
            errors,
        )

    self_value = {
        "files": files,
        "bodies": bodies,
        "metadata": metadata,
    }
    try:
        self_digest = sha256_json(self_value)
    except (TypeError, ValueError):
        errors.append("manifest sha256 cannot be fingerprinted as JSON")
    else:
        verify_digest("self", hashes.get("self"), self_digest, errors)

    if set(bodies) != {_BODY_NAME}:
        errors.append("object manifest sha256 bodies must contain exactly 'object'")
    digest = bodies.get(_BODY_NAME)
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        errors.append("manifest body 'object' has an invalid geometry SHA-256")
    for relative in files:
        if isinstance(relative, str) and Path(relative).parts[:1] == ("collision",):
            errors.append(
                "manifest sha256 files must not fingerprint collision pieces "
                f"individually: {relative}"
            )
    return files, bodies


def _check_manifest_relations(
    root: Path,
    manifest: dict[str, object],
    files: dict[object, object],
    bodies: dict[object, object],
    errors: list[str],
) -> None:
    surfaces = manifest.get("surfaces")
    if isinstance(surfaces, dict):
        if set(surfaces) != {_BODY_NAME}:
            errors.append("object manifest surfaces must contain exactly 'object'")
        relative = surfaces.get(_BODY_NAME)
        if not isinstance(relative, str):
            errors.append("object surface path must be a string")
        else:
            _check_artifact_reference(
                root,
                relative,
                files,
                "object surface",
                errors,
            )

    models = manifest.get("models")
    if not isinstance(models, dict):
        return
    if not models:
        errors.append("object manifest models must not be empty")
    for relative, model_bodies in models.items():
        if not isinstance(relative, str) or not relative:
            errors.append("manifest model paths must be non-empty strings")
            continue
        _check_artifact_reference(
            root,
            relative,
            files,
            f"model {relative!r}",
            errors,
        )
        if model_bodies != [_BODY_NAME]:
            errors.append(
                f"object manifest model must contain exactly body 'object': {relative!r}"
            )


def _check_artifact_reference(
    root: Path,
    relative: str,
    files: dict[object, object],
    description: str,
    errors: list[str],
) -> None:
    if relative in files:
        return
    errors.append(f"{description} is not fingerprinted: {relative}")
    try:
        path = resolve_artifact(root, relative)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if not path.is_file():
            errors.append(f"{description} is missing: {relative}")


def _check_object_meshes(
    root: Path,
    manifest: dict[str, object],
    bodies: dict[object, object],
    errors: list[str],
) -> None:
    import trimesh

    from ..mesh.fingerprint import compiled_collision_geometry_sha256
    from ..mesh.io import load_mesh
    from ..mesh.properties import collision_properties
    from ..mesh.validation import validate_mesh

    surfaces = manifest.get("surfaces")
    if isinstance(surfaces, dict) and isinstance(surfaces.get(_BODY_NAME), str):
        try:
            visual_path = resolve_artifact(root, surfaces[_BODY_NAME])
        except ValueError:
            pass
        else:
            if visual_path.is_file():
                _load_and_validate_mesh(
                    visual_path,
                    "visual",
                    load_mesh,
                    validate_mesh,
                    errors,
                )

    collision_paths = _collision_paths(root, errors)
    collision_meshes = []
    for path in collision_paths:
        mesh = _load_and_validate_mesh(
            path,
            f"collision mesh {path.relative_to(root).as_posix()}",
            load_mesh,
            validate_mesh,
            errors,
        )
        if mesh is not None:
            collision_meshes.append(mesh)

    if collision_meshes and len(collision_meshes) == len(collision_paths):
        try:
            actual = compiled_collision_geometry_sha256(collision_paths)
        except (RuntimeError, ValueError) as exc:
            errors.append(f"could not fingerprint object collision geometry: {exc}")
        else:
            verify_digest(
                "body object",
                bodies.get(_BODY_NAME),
                actual,
                errors,
            )
        try:
            mass_properties = collision_properties(
                trimesh.util.concatenate(collision_meshes)
            )
        except ValueError as exc:
            errors.append(f"could not compute object mass properties: {exc}")
            mass_properties = None
    else:
        mass_properties = None

    expected_collision_paths = Counter(
        path.relative_to(root).as_posix() for path in collision_paths
    )
    expected_surface = (
        surfaces.get(_BODY_NAME)
        if isinstance(surfaces, dict) and isinstance(surfaces.get(_BODY_NAME), str)
        else None
    )
    models = manifest.get("models")
    if not isinstance(models, dict):
        return
    for relative in models:
        if not isinstance(relative, str):
            continue
        try:
            model_path = resolve_artifact(root, relative)
        except ValueError:
            continue
        if not model_path.is_file():
            continue
        references = _model_collision_paths(
            root,
            model_path,
            bodies.get(_BODY_NAME),
            mass_properties,
            expected_surface,
            errors,
        )
        if references is not None and references != expected_collision_paths:
            errors.append(
                f"model collision geometry does not match the object bundle: {relative}"
            )


def _collision_paths(root: Path, errors: list[str]) -> list[Path]:
    directory = root / "collision"
    if directory.is_symlink():
        errors.append("collision directory must not be a symbolic link")
        return []
    if not directory.is_dir():
        errors.append("collision directory is missing")
        return []

    paths = []
    for path in sorted(directory.iterdir()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"unexpected collision symbolic link: {relative}")
        elif not path.is_file() or path.suffix.lower() != ".obj":
            errors.append(f"unexpected collision artifact: {relative}")
        else:
            paths.append(path)
    if not paths:
        errors.append("collision directory must contain at least one OBJ mesh")
    return paths


def _load_and_validate_mesh(
    path: Path,
    label: str,
    load_mesh,
    validate_mesh,
    errors: list[str],
):
    try:
        mesh = load_mesh(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"{label} could not be loaded: {exc}")
        return None
    mesh_errors = validate_mesh(mesh, watertight=True)
    for error in mesh_errors:
        errors.append(f"{label} is invalid: {error}")
    return None if mesh_errors else mesh


def _model_collision_paths(
    bundle_root: Path,
    model_path: Path,
    expected_body_digest: object,
    mass_properties: dict[str, object] | None,
    expected_surface: str | None,
    errors: list[str],
) -> Counter[str] | None:
    try:
        root = ET.parse(model_path).getroot()
    except (OSError, ET.ParseError) as exc:
        errors.append(
            f"model could not be parsed: {model_path.relative_to(bundle_root)}: {exc}"
        )
        return None
    if root.tag == "mujoco":
        return _mjcf_collision_paths(
            bundle_root,
            model_path,
            root,
            expected_body_digest,
            mass_properties,
            expected_surface,
            errors,
        )
    if root.tag == "robot":
        return _urdf_collision_paths(
            bundle_root,
            model_path,
            root,
            expected_surface,
            errors,
        )
    errors.append(
        f"object model must be MJCF or URDF: {model_path.relative_to(bundle_root)}"
    )
    return None


def _mjcf_collision_paths(
    bundle_root: Path,
    model_path: Path,
    root: ET.Element,
    expected_body_digest: object,
    mass_properties: dict[str, object] | None,
    expected_surface: str | None,
    errors: list[str],
) -> Counter[str] | None:
    label = model_path.relative_to(bundle_root).as_posix()
    compiler = root.find("./compiler")
    mesh_directory = compiler.get("meshdir", ".") if compiler is not None else "."
    assets: dict[str, str] = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name")
        filename = mesh.get("file")
        if not name or not filename:
            errors.append(f"MJCF mesh assets require names and files: {label}")
        elif name in assets:
            errors.append(f"MJCF mesh asset names must be unique: {label}")
        else:
            assets[name] = filename
        if mesh.get("scale") is not None:
            errors.append(f"MJCF object mesh assets must not specify scale: {label}")

    bodies = root.findall(".//body")
    if len(bodies) != 1 or bodies[0].get("name") != _BODY_NAME:
        errors.append(f"MJCF model must contain only one body named 'object': {label}")
        return None
    body = bodies[0]
    worldbody = root.find("./worldbody")
    if worldbody is None or body not in list(worldbody):
        errors.append(f"MJCF object body must be in worldbody: {label}")
        return None
    if _MJCF_POSE_ATTRIBUTES.intersection(body.attrib):
        errors.append(f"MJCF object body must use an identity pose: {label}")
    _check_mjcf_inertial(body, label, mass_properties, errors)

    references: Counter[str] = Counter()
    visual_references: list[str] = []
    for geom in body.findall("geom"):
        try:
            contype = int(geom.get("contype", "1"), 0)
            conaffinity = int(geom.get("conaffinity", "1"), 0)
        except ValueError:
            errors.append(f"MJCF geom has invalid collision masks: {label}")
            continue
        if geom.get("type", "sphere") != "mesh" or not geom.get("mesh"):
            errors.append(f"MJCF object geoms must all be meshes: {label}")
            continue
        if _MJCF_POSE_ATTRIBUTES.intersection(geom.attrib):
            errors.append(f"MJCF object geoms must use identity poses: {label}")
        mesh_name = geom.get("mesh")
        filename = assets.get(mesh_name)
        if filename is None:
            errors.append(f"MJCF geom references a missing mesh: {label}")
            continue
        relative = _model_asset_path(
            bundle_root,
            model_path.parent,
            str(PurePosixPath(mesh_directory) / filename),
            errors,
        )
        if relative is None:
            continue
        if contype == 0 and conaffinity == 0:
            visual_references.append(relative)
        else:
            references[relative] += 1

    if visual_references != [expected_surface]:
        errors.append(f"MJCF visual mesh does not match the manifest surface: {label}")
    if root.find("./contact/pair") is not None:
        errors.append(
            f"MJCF object bundles must not use explicit contact pairs: {label}"
        )
    try:
        from ..mesh.fingerprint import mujoco_body_geometry_sha256

        actual = mujoco_body_geometry_sha256(model_path)
    except (RuntimeError, ValueError) as exc:
        errors.append(
            f"MJCF object collision geometry could not be compiled: {label}: {exc}"
        )
    else:
        verify_digest(
            f"model body object ({label})",
            expected_body_digest,
            actual,
            errors,
        )
    return references


def _check_mjcf_inertial(
    body: ET.Element,
    label: str,
    mass_properties: dict[str, object] | None,
    errors: list[str],
) -> None:
    inertial = body.find("inertial")
    if inertial is None:
        errors.append(f"MJCF object body must contain an explicit inertial: {label}")
        return
    position = _numeric_vector(inertial.get("pos"), 3)
    full = _numeric_vector(inertial.get("fullinertia"), 6)
    try:
        mass = float(inertial.get("mass", ""))
    except ValueError:
        mass = math.nan
    if position is None:
        errors.append(f"MJCF object inertial pos must be a finite 3-vector: {label}")
    if not math.isfinite(mass) or mass <= 0:
        errors.append(f"MJCF object inertial mass must be positive and finite: {label}")
    if full is None:
        errors.append(f"MJCF object fullinertia must be a finite 6-vector: {label}")
        return
    diagonal = full[:3]
    if any(value <= 0 for value in diagonal):
        errors.append(f"MJCF object inertia must have positive diagonal terms: {label}")
    if mass_properties is None or position is None or not math.isfinite(mass):
        return

    import numpy as np

    from .mjcf import object_reference_inertial

    expected_mass, expected_position, expected_full = object_reference_inertial(
        mass_properties
    )
    if not math.isclose(mass, expected_mass, rel_tol=1.0e-9, abs_tol=1.0e-12):
        errors.append(
            f"MJCF object inertial mass does not match collision geometry: {label}"
        )
    if not np.allclose(
        position,
        expected_position,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        errors.append(
            f"MJCF object inertial pos does not match collision geometry: {label}"
        )
    if not np.allclose(
        full,
        expected_full,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        errors.append(
            f"MJCF object fullinertia does not match collision geometry: {label}"
        )


def _urdf_collision_paths(
    bundle_root: Path,
    model_path: Path,
    root: ET.Element,
    expected_surface: str | None,
    errors: list[str],
) -> Counter[str] | None:
    label = model_path.relative_to(bundle_root).as_posix()
    links = root.findall("link")
    if len(links) != 1 or links[0].get("name") != _BODY_NAME or list(root) != links:
        errors.append(f"URDF model must contain only one link named 'object': {label}")
        return None
    link = links[0]

    visual_nodes = link.findall("visual")
    visual_references = []
    for visual in visual_nodes:
        relative = _urdf_mesh_path(
            bundle_root,
            model_path,
            visual,
            "visual",
            errors,
        )
        if relative is not None:
            visual_references.append(relative)
    if visual_references != [expected_surface]:
        errors.append(f"URDF visual mesh does not match the manifest surface: {label}")

    references: Counter[str] = Counter()
    for collision in link.findall("collision"):
        relative = _urdf_mesh_path(
            bundle_root,
            model_path,
            collision,
            "collision",
            errors,
        )
        if relative is not None:
            references[relative] += 1
    return references


def _urdf_mesh_path(
    bundle_root: Path,
    model_path: Path,
    record: ET.Element,
    kind: str,
    errors: list[str],
) -> str | None:
    label = model_path.relative_to(bundle_root).as_posix()
    if record.attrib:
        errors.append(f"URDF object {kind} record must not have attributes: {label}")
    record_children = list(record)
    geometry = record_children[0] if record_children else None
    if (
        len(record_children) != 1
        or geometry is None
        or geometry.tag != "geometry"
        or geometry.attrib
        or len(geometry) != 1
        or geometry[0].tag != "mesh"
        or not geometry[0].get("filename")
    ):
        errors.append(
            f"URDF object {kind} record must contain exactly one geometry and one mesh: {label}"
        )
        return None
    mesh = geometry[0]
    if mesh.get("scale") is not None:
        errors.append(f"URDF object meshes must not specify scale: {label}")
    return _model_asset_path(
        bundle_root,
        model_path.parent,
        mesh.get("filename"),
        errors,
    )

def _model_asset_path(
    bundle_root: Path,
    directory: Path,
    value: str,
    errors: list[str],
) -> str | None:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        errors.append(f"model asset path must be portable and relative: {value!r}")
        return None
    path = (directory / value).resolve()
    try:
        relative = path.relative_to(bundle_root.resolve()).as_posix()
    except ValueError:
        errors.append(f"model asset path escapes the object bundle: {value!r}")
        return None
    cursor = bundle_root
    for part in Path(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            errors.append(f"model asset path must not traverse symlinks: {value!r}")
            return None
    return relative


def _numeric_vector(value: str | None, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(component) for component in value.split()]
    except ValueError:
        return None
    if len(result) != length or not all(
        math.isfinite(component) for component in result
    ):
        return None
    return result


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_numeric_array(value: object, shape: tuple[int, ...]) -> bool:
    if not shape:
        return _is_finite_number(value)
    if not isinstance(value, list) or len(value) != shape[0]:
        return False
    return all(_is_numeric_array(item, shape[1:]) for item in value)
