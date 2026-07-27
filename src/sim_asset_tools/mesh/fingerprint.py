"""Exact fingerprints for source body-local collision meshes."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable

_BODY_GEOMETRY_DOMAIN = b"source-body-triangle-mesh-sha256-v1\0"


def body_geometry_sha256(meshes: Iterable[object]) -> str:
    """Hash source triangle meshes after concatenating them in body coordinates.

    The fingerprint preserves the source vertex and face representation. Callers
    must provide meshes in a deterministic order with all authored body-local
    transforms already applied.
    """
    import numpy as np

    vertex_chunks = []
    face_chunks = []
    vertex_offset = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype="<f8")
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
            vertex_chunks.append(np.ascontiguousarray(vertices, dtype="<f8"))
            indexed_faces = np.asarray(faces, dtype="<u8") + vertex_offset
            face_chunks.append(np.ascontiguousarray(indexed_faces, dtype="<u8"))
            vertex_offset += len(vertices)

    if not face_chunks:
        raise ValueError("Body geometry must contain at least one triangle")

    vertices = np.ascontiguousarray(np.concatenate(vertex_chunks), dtype="<f8")
    faces = np.ascontiguousarray(np.concatenate(face_chunks), dtype="<u8")
    digest = hashlib.sha256()
    digest.update(_BODY_GEOMETRY_DOMAIN)
    digest.update(struct.pack("<QQ", len(vertices), len(faces)))
    digest.update(vertices.tobytes(order="C"))
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()
