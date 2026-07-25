"""Prepare self-describing visual, collision, MJCF, and URDF object assets."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..formats.manifest import MANIFEST_NAME
from ..formats.object_manifest import write_object_manifest
from ..formats.mjcf import write_object_mjcf
from ..formats.urdf import write_object_urdf
from ..mesh.coacd import decompose_mesh
from ..mesh.fingerprint import compiled_collision_geometry_sha256
from ..mesh.io import SUPPORTED_MESH_SUFFIXES, load_mesh
from ..mesh.normalize import normalize_mesh
from ..mesh.properties import collision_properties, oriented_bounding_box
from ..mesh.validation import validate_mesh
from ..publish import ensure_safe_output, staged_directory
from ..surface import SurfaceRecipe, prepare_surface


@dataclass(frozen=True)
class ObjectRecipe:
    """Deterministic object preparation settings."""

    normalize: bool = True
    resolution: float = 50.0
    level_set: float = 0.1
    target_vertices: int = 1024
    gradation: float = 1.5
    force_manifold: int = 1
    coacd_threshold: float = 0.05
    coacd_max_convex_hull: int = -1
    coacd_preprocess_mode: str = "auto"
    coacd_preprocess_resolution: int = 50
    coacd_real_metric: bool = False
    seed: int | None = 0

    @property
    def surface(self) -> SurfaceRecipe:
        return SurfaceRecipe(
            resolution=self.resolution,
            level_set=self.level_set,
            target_vertices=self.target_vertices,
            gradation=self.gradation,
            force_manifold=self.force_manifold,
        )


@dataclass(frozen=True)
class ObjectResult:
    """Paths produced for one prepared object."""

    output_directory: Path
    manifest_path: Path
    mjcf_path: Path | None
    urdf_path: Path | None


def prepare_object(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    recipe: ObjectRecipe | None = None,
    formats: tuple[str, ...] = ("mjcf", "urdf"),
    overwrite: bool = False,
) -> ObjectResult:
    """Prepare one input mesh as a self-describing simulation object bundle."""
    import trimesh

    input_path = Path(input_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    recipe = recipe or ObjectRecipe()
    formats = tuple(dict.fromkeys(formats))
    if not formats:
        raise ValueError("At least one object output format is required")
    unsupported_formats = sorted(set(formats) - {"mjcf", "urdf"})
    if unsupported_formats:
        raise ValueError(
            f"Unsupported object output formats: {', '.join(unsupported_formats)}"
        )
    if not input_path.is_file():
        raise ValueError(f"Input mesh does not exist: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(f"Unsupported input mesh suffix: {input_path.suffix}")
    ensure_safe_output(input_path, output_directory)

    with staged_directory(output_directory, overwrite=overwrite) as staging_directory:
        collision_directory = staging_directory / "collision"
        collision_directory.mkdir(parents=True)

        source_copy = staging_directory / f"source{input_path.suffix.lower()}"
        shutil.copy2(input_path, source_copy)
        mesh = load_mesh(input_path)
        normalized_mesh, source_center, _ = normalize_mesh(mesh)
        source_extents = mesh.extents.tolist()
        if recipe.normalize:
            processing_mesh = normalized_mesh
        else:
            processing_mesh = mesh.copy()

        with tempfile.TemporaryDirectory(
            prefix=".work-", dir=staging_directory
        ) as work_value:
            work_directory = Path(work_value)
            processing_source = work_directory / "source.obj"
            processing_mesh.export(processing_source)
            acvd_output = work_directory / "simplified.ply"
            prepare_surface(
                processing_source,
                acvd_output,
                work_directory / "surface",
                recipe.surface,
            )
            visual_mesh = load_mesh(acvd_output)
            visual_path = staging_directory / "visual.obj"
            visual_mesh.export(visual_path)
            visual_mesh = load_mesh(visual_path)
            visual_errors = validate_mesh(visual_mesh, watertight=True)
            if visual_errors:
                raise ValueError(
                    f"Published visual mesh is invalid: {'; '.join(visual_errors)}"
                )

        decomposed_meshes = decompose_mesh(
            visual_mesh,
            threshold=recipe.coacd_threshold,
            max_convex_hull=recipe.coacd_max_convex_hull,
            preprocess_mode=recipe.coacd_preprocess_mode,
            preprocess_resolution=recipe.coacd_preprocess_resolution,
            real_metric=recipe.coacd_real_metric,
            seed=recipe.seed,
        )
        if not decomposed_meshes:
            raise RuntimeError("CoACD did not produce any collision parts")
        collision_paths: list[Path] = []
        collision_meshes = []
        for index, collision_mesh in enumerate(decomposed_meshes):
            collision_path = collision_directory / f"part_{index:03d}.obj"
            collision_mesh.export(collision_path)
            collision_mesh = load_mesh(collision_path)
            errors = validate_mesh(collision_mesh, watertight=True)
            if errors:
                raise ValueError(f"CoACD part {index} is invalid: {'; '.join(errors)}")
            collision_paths.append(collision_path)
            collision_meshes.append(collision_mesh)

        combined_collision = trimesh.util.concatenate(collision_meshes)
        mass_properties = collision_properties(combined_collision)
        mjcf_path = staging_directory / "model.xml" if "mjcf" in formats else None
        urdf_path = staging_directory / "model.urdf" if "urdf" in formats else None
        if mjcf_path is not None:
            write_object_mjcf(
                mjcf_path,
                visual_path,
                collision_paths,
                mass_properties,
            )
        if urdf_path is not None:
            write_object_urdf(urdf_path, visual_path, collision_paths)
        body_sha256 = compiled_collision_geometry_sha256(collision_paths)
        artifacts = [source_copy, visual_path]
        models = [path for path in (mjcf_path, urdf_path) if path is not None]
        artifacts.extend(models)
        write_object_manifest(
            staging_directory / MANIFEST_NAME,
            source_aabb_center=source_center,
            source_aabb_extents=source_extents,
            obb=oriented_bounding_box(visual_mesh),
            recipe=asdict(recipe),
            surface=visual_path,
            models=models,
            artifacts=artifacts,
            body_sha256=body_sha256,
        )

    return ObjectResult(
        output_directory,
        output_directory / MANIFEST_NAME,
        output_directory / "model.xml" if "mjcf" in formats else None,
        output_directory / "model.urdf" if "urdf" in formats else None,
    )
