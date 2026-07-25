"""Canonical fingerprints for complete body-local collision geometry."""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET

_BODY_GEOMETRY_DOMAIN = b"body-geometry-sha256-v1\0"
_BODY_GEOMETRY_QUANTIZATION = 1.0e-6


def body_geometry_sha256(meshes: Iterable[object]) -> str:
    """Hash a body-local triangle soup independently of mesh bookkeeping.

    Vertices are first represented as float32, matching compiled simulation
    geometry, then quantized to micrometers. Vertex order, triangle winding,
    face order, mesh grouping, and mesh filenames do not affect the result.
    Duplicate triangles remain significant.
    """
    import numpy as np

    triangles = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
        ):
            raise ValueError("Body geometry must contain triangular meshes")
        if not np.isfinite(vertices).all():
            raise ValueError("Body geometry vertices must be finite")
        if faces.size and (
            not np.issubdtype(faces.dtype, np.integer)
            or int(faces.min()) < 0
            or int(faces.max()) >= len(vertices)
        ):
            raise ValueError("Body geometry faces reference invalid vertices")
        if faces.size:
            triangles.append(vertices[faces])

    if not triangles:
        raise ValueError("Body geometry must contain at least one triangle")

    values = np.concatenate(triangles, axis=0).astype(np.float64)
    rounded = np.rint(values / _BODY_GEOMETRY_QUANTIZATION)
    if bool(np.any(rounded < -(2**63))) or bool(np.any(rounded >= 2**63)):
        raise ValueError("Body geometry exceeds the fingerprint coordinate range")
    quantized = rounded.astype("<i8")

    vertex_order = np.lexsort(
        (
            quantized[:, :, 2],
            quantized[:, :, 1],
            quantized[:, :, 0],
        ),
        axis=1,
    )
    canonical = np.take_along_axis(
        quantized,
        vertex_order[:, :, None],
        axis=1,
    ).reshape((-1, 9))
    triangle_order = np.lexsort(canonical[:, ::-1].T)
    canonical = np.ascontiguousarray(canonical[triangle_order], dtype="<i8")

    digest = hashlib.sha256()
    digest.update(_BODY_GEOMETRY_DOMAIN)
    digest.update(struct.pack("<Q", len(canonical)))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def mujoco_body_geometry_sha256(
    model_or_path,
    body_name: str = "object",
) -> str:
    """Hash one compiled MuJoCo body's complete collidable geometry."""
    mujoco = _import_mujoco()

    if isinstance(model_or_path, mujoco.MjModel):
        model = model_or_path
    else:
        path = Path(model_or_path)
        if path.suffix.lower() == ".mjb":
            model = mujoco.MjModel.from_binary_path(path.as_posix())
        else:
            model = mujoco.MjSpec.from_file(path.as_posix()).compile()

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if body_id < 0:
        raise ValueError(f"MuJoCo model has no body named {body_name!r}")
    pair_geoms = {
        int(geom_id)
        for pair_id in range(model.npair)
        for geom_id in (
            model.pair_geom1[pair_id],
            model.pair_geom2[pair_id],
        )
    }
    meshes = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
            and geom_id not in pair_geoms
        ):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(
                f"MuJoCo body {body_name!r} has a non-mesh collidable geom"
            )
        meshes.append(_mesh_geom_from_mujoco(model, geom_id))
    return body_geometry_sha256(meshes)


def compiled_collision_geometry_sha256(paths: Iterable[Path]) -> str:
    """Compile collision mesh files with MuJoCo and hash their body geometry."""
    mujoco = _import_mujoco()

    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError("Body geometry must contain at least one collision mesh")
    root = ET.Element("mujoco", {"model": "body_geometry_fingerprint"})
    assets = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "object"})
    asset_contents = {}
    for index, path in enumerate(paths):
        name = f"collision_{index:06d}"
        filename = f"{name}{path.suffix.lower()}"
        ET.SubElement(
            assets,
            "mesh",
            {"name": name, "file": filename},
        )
        ET.SubElement(
            body,
            "geom",
            {"name": name, "type": "mesh", "mesh": name},
        )
        asset_contents[filename] = path.read_bytes()
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"),
        assets=asset_contents,
    )
    return mujoco_body_geometry_sha256(model)


def _import_mujoco():
    os.environ.setdefault("MUJOCO_GL", "disable")
    import mujoco

    return mujoco


def _mesh_geom_from_mujoco(model, geom_id: int):
    import numpy as np
    import trimesh

    mesh_id = int(model.geom_dataid[geom_id])
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    polygon_address = int(model.mesh_polyadr[mesh_id])
    polygon_count = int(model.mesh_polynum[mesh_id])
    source_vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=np.float32,
    )
    faces = []
    for polygon_id in range(
        polygon_address,
        polygon_address + polygon_count,
    ):
        start = int(model.mesh_polyvertadr[polygon_id])
        count = int(model.mesh_polyvertnum[polygon_id])
        if count >= 3:
            polygon = np.asarray(
                model.mesh_polyvert[start : start + count],
                dtype=np.int64,
            )
            faces.extend(_canonical_polygon_triangles(source_vertices, polygon))
    faces = np.asarray(faces, dtype=np.int64)
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = source_vertices[used_vertices]
    faces = inverse.reshape((-1, 3)).astype(np.int32)

    rotation = _quaternion_matrix(
        np.asarray(model.geom_quat[geom_id], dtype=np.float32)
    )
    position = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
    return trimesh.Trimesh(
        vertices=vertices @ rotation.T + position,
        faces=faces,
        process=False,
    )


def _canonical_polygon_triangles(vertices, polygon):
    """Triangulate a polygon independently of its cyclic start and winding."""
    anchor = min(
        range(len(polygon)),
        key=lambda offset: tuple(
            float(component) for component in vertices[polygon[offset]]
        ),
    )
    canonical = tuple(
        polygon[(anchor + offset) % len(polygon)] for offset in range(len(polygon))
    )
    return [
        (int(canonical[0]), int(canonical[index]), int(canonical[index + 1]))
        for index in range(1, len(canonical) - 1)
    ]


def _quaternion_matrix(quaternion):
    import numpy as np

    w, x, y, z = quaternion
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )
