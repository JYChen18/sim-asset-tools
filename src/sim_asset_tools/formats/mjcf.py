"""MJCF output helpers."""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree as ET

OBJECT_REFERENCE_DENSITY = 1000.0


def _relative_path(owner: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=owner.parent)).as_posix()


def _numbers(values) -> str:
    """Format numeric XML attributes without discarding useful precision."""
    return " ".join(format(float(value), ".17g") for value in values)


def object_reference_inertial(
    mass_properties: dict[str, object],
) -> tuple[float, object, object]:
    """Return reference mass, center, and full inertia for an object model."""
    import numpy as np

    mass = float(mass_properties["volume"]) * OBJECT_REFERENCE_DENSITY
    center = np.asarray(mass_properties["center_of_mass"], dtype=float)
    inertia = np.asarray(mass_properties["inertia_per_unit_mass"], dtype=float) * mass
    full_inertia = inertia[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    return mass, center, full_inertia


def write_object_mjcf(
    path: Path,
    visual: Path,
    collisions: list[Path],
    mass_properties: dict[str, object],
) -> None:
    """Write a one-body MJCF model referencing prepared object meshes."""
    mass, center, full_inertia = object_reference_inertial(mass_properties)

    root = ET.Element("mujoco", {"model": "sim_asset"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "mesh",
        {"name": "visual_mesh", "file": _relative_path(path, visual)},
    )
    for index, collision in enumerate(collisions):
        ET.SubElement(
            assets,
            "mesh",
            {
                "name": f"collision_{index:03d}",
                "file": _relative_path(path, collision),
            },
        )
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "object"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": _numbers(center),
            "mass": format(mass, ".17g"),
            "fullinertia": _numbers(full_inertia),
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "visual",
            "type": "mesh",
            "mesh": "visual_mesh",
            "density": "0",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for index in range(len(collisions)):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"collision_{index:03d}",
                "type": "mesh",
                "mesh": f"collision_{index:03d}",
                "density": "0",
            },
        )
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    with path.open("a", encoding="utf-8") as destination:
        destination.write("\n")
