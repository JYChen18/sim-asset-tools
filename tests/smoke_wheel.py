from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sim_asset_tools.cli import main as sim_assets_main
from sim_asset_tools.native import _native_env, _tool_path, run_native


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tetrahedron.obj"
EXPECTED_NATIVE_TOOLS = {
    "ACVD",
    "ACVDP",
    "ACVDQ",
    "ACVDQP",
    "AnisotropicRemeshing",
    "AnisotropicRemeshingQ",
    "AnisotropicRemeshingQP",
    "OpenVDBSdfRemesh",
    "VolumeAnalysis",
}
ACVD_CASES = (
    ("acvd", "ACVD", (), ()),
    ("acvd-parallel", "ACVDP", ("--threads", "1"), ("-np", "1")),
    ("acvd-quadric", "ACVDQ", ("--quadric-level", "1"), ("-q", "1")),
    (
        "acvd-quadric-parallel",
        "ACVDQP",
        ("--quadric-level", "1", "--threads", "1"),
        ("-q", "1", "-np", "1"),
    ),
    ("acvd-anisotropic", "AnisotropicRemeshing", (), ()),
    (
        "acvd-anisotropic-quadric",
        "AnisotropicRemeshingQ",
        ("--quadric-level", "1"),
        ("-q", "1"),
    ),
    (
        "acvd-anisotropic-quadric-parallel",
        "AnisotropicRemeshingQP",
        ("--quadric-level", "1", "--threads", "1"),
        ("-q", "1", "-np", "1"),
    ),
)


def require_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Expected non-empty output: {path}")


def diagnose_native_crash(tool: str, native_args: list[str]) -> None:
    gdb = shutil.which("gdb")
    if not sys.platform.startswith("linux") or gdb is None:
        return

    print(f"{tool} terminated by a signal; rerunning under gdb", file=sys.stderr)
    subprocess.run(
        [
            gdb,
            "--batch",
            "-ex",
            "set pagination off",
            "-ex",
            "run",
            "-ex",
            "thread apply all backtrace",
            "--args",
            str(_tool_path(tool)),
            *native_args,
        ],
        check=False,
        env=_native_env(),
        text=True,
    )


def verify_installed_tools() -> None:
    for tool in sorted(EXPECTED_NATIVE_TOOLS):
        path = _tool_path(tool)
        if not path.is_file():
            raise RuntimeError(f"Expected installed native executable: {path}")


def verify_cli_entry_point() -> None:
    executable = shutil.which("sim-assets")
    if executable is None:
        raise RuntimeError("The installed sim-assets entry point is missing")
    subprocess.run(
        [executable, "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
        text=True,
    )


def run_normalize(temporary_path: Path) -> Path:
    output = temporary_path / "normalize" / "object.obj"
    result = sim_assets_main(["mesh", "normalize", str(FIXTURE), str(output)])
    if result != 0:
        raise RuntimeError(f"Normalize smoke test failed with exit code {result}")
    require_output(output)
    return output


def run_openvdb(input_path: Path, temporary_path: Path) -> Path:
    output = temporary_path / "openvdb" / "object.obj"
    result = sim_assets_main(
        [
            "mesh",
            "openvdb",
            str(input_path),
            str(output),
            "--resolution",
            "10",
        ]
    )
    if result != 0:
        raise RuntimeError(f"OpenVDB smoke test failed with exit code {result}")
    require_output(output)
    return output


def run_acvd_cases(temporary_path: Path, default_input: Path) -> Path:
    default_output = None
    for method, tool, cli_extra, native_extra in ACVD_CASES:
        input_path = default_input if method == "acvd" else FIXTURE
        output = temporary_path / f"{method}.ply"
        argv = [
            "mesh",
            "acvd",
            str(input_path),
            str(output),
            "--method",
            method,
            "--vertices",
            "4",
            "--gradation",
            "0",
            "--subsample",
            "1",
            *cli_extra,
        ]
        result = sim_assets_main(argv)
        if result != 0:
            if result < 0:
                native_args = [
                    str(input_path),
                    "4",
                    "0",
                    "-o",
                    f"{temporary_path}{os.sep}",
                ]
                if not tool.startswith("AnisotropicRemeshing"):
                    native_args.extend(("-of", output.name))
                native_args.extend(("-s", "1", *native_extra))
                diagnose_native_crash(tool, native_args)
            raise RuntimeError(f"{method} smoke test failed with exit code {result}")
        require_output(output)
        if method == "acvd":
            default_output = output

    dotted_output = temporary_path / ".object" / "output.ply"
    result = sim_assets_main(
        [
            "mesh",
            "acvd",
            str(FIXTURE),
            str(dotted_output),
            "--vertices",
            "4",
            "--gradation",
            "0",
            "--subsample",
            "1",
        ]
    )
    if result != 0:
        raise RuntimeError(f"ACVD dotted-path test failed with exit code {result}")
    require_output(dotted_output)
    if not dotted_output.read_bytes().startswith(b"ply"):
        raise RuntimeError(f"ACVD wrote the wrong format: {dotted_output}")
    if default_output is None:
        raise RuntimeError("Default ACVD smoke test did not run")
    return default_output


def write_volume_fixture(root: Path) -> Path:
    raw_path = root / "labels.raw"
    values = bytearray(5 * 5 * 5)
    for z in range(1, 4):
        for y in range(1, 4):
            for x in range(1, 4):
                values[x + 5 * (y + 5 * z)] = 1
    raw_path.write_bytes(values)

    header_path = root / "labels.mhd"
    header_path.write_text(
        "\n".join(
            (
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                "Offset = 0 0 0",
                "CenterOfRotation = 0 0 0",
                "ElementSpacing = 1 1 1",
                "DimSize = 5 5 5",
                "ElementType = MET_UCHAR",
                f"ElementDataFile = {raw_path.name}",
                "",
            )
        ),
        encoding="ascii",
    )
    return header_path


def run_volume_analysis(temporary_path: Path) -> None:
    fixture = write_volume_fixture(temporary_path)
    output_directory = temporary_path / "volume-output"
    output_directory.mkdir()
    native_args = [
        str(fixture),
        "-n",
        "0",
        "-j",
        "1",
        "-f",
        "ply",
        "-o",
        str(output_directory),
    ]
    result = run_native(
        "VolumeAnalysis",
        native_args,
        cwd=temporary_path,
    )
    if result.returncode < 0:
        diagnose_native_crash("VolumeAnalysis", native_args)
    if result.returncode != 0:
        raise RuntimeError(
            f"VolumeAnalysis smoke test failed with exit code {result.returncode}"
        )
    require_output(output_directory / "1.ply")
    require_output(temporary_path / "meshes.xml")


def run_coacd(input_path: Path, temporary_path: Path) -> None:
    output_directory = temporary_path / "coacd"
    result = sim_assets_main(
        ["mesh", "coacd", str(input_path), str(output_directory)]
    )
    if result != 0:
        raise RuntimeError(f"CoACD smoke test failed with exit code {result}")
    parts = sorted(output_directory.glob("part_*.obj"))
    if not parts:
        raise RuntimeError("CoACD smoke test did not produce collision parts")
    for part in parts:
        require_output(part)


def object_recipe_arguments() -> list[str]:
    return [
        "--resolution",
        "10",
        "--vertices",
        "64",
        "--gradation",
        "0",
        "--coacd-preprocess-resolution",
        "10",
    ]


def verify_object_bundle(output_directory: Path) -> None:
    for relative_path in (
        "asset.json",
        "visual.obj",
        "model.xml",
        "model.urdf",
    ):
        require_output(output_directory / relative_path)
    collision_parts = sorted((output_directory / "collision").glob("part_*.obj"))
    if not collision_parts:
        raise RuntimeError(
            f"Prepared object has no collision parts: {output_directory}"
        )
    for part in collision_parts:
        require_output(part)

    result = sim_assets_main(["check", "object", str(output_directory)])
    if result != 0:
        raise RuntimeError(
            f"Object validation smoke test failed with exit code {result}: "
            f"{output_directory}"
        )


def run_prepare_object(temporary_path: Path) -> None:
    output_directory = temporary_path / "prepare-object"
    result = sim_assets_main(
        [
            "prepare",
            "object",
            str(FIXTURE),
            "--output",
            str(output_directory),
            *object_recipe_arguments(),
        ]
    )
    if result != 0:
        raise RuntimeError(
            f"Single-object preparation smoke test failed with exit code {result}"
        )
    verify_object_bundle(output_directory)


def run_prepare_objects(temporary_path: Path) -> None:
    input_directory = temporary_path / "objects"
    input_directory.mkdir()
    for name in ("first.obj", "second.obj"):
        shutil.copy2(FIXTURE, input_directory / name)

    output_directory = temporary_path / "prepare-objects"
    result = sim_assets_main(
        [
            "prepare",
            "objects",
            str(input_directory),
            "--output",
            str(output_directory),
            "--jobs",
            "2",
            *object_recipe_arguments(),
        ]
    )
    if result != 0:
        raise RuntimeError(
            f"Batch object preparation smoke test failed with exit code {result}"
        )
    for name in ("first", "second"):
        verify_object_bundle(output_directory / name)


def run_public_workflow_primitives(temporary_path: Path) -> None:
    from sim_asset_tools.publish import staged_directory
    from sim_asset_tools.surface import SurfaceRecipe

    recipe = SurfaceRecipe(target_vertices=64)
    if recipe.target_vertices != 64:
        raise RuntimeError("SurfaceRecipe smoke test returned unexpected settings")

    output_directory = temporary_path / "published"
    with staged_directory(output_directory, overwrite=False) as staging_directory:
        (staging_directory / "ready.txt").write_text("ready\n", encoding="utf-8")
    require_output(output_directory / "ready.txt")


def run() -> None:
    verify_installed_tools()
    verify_cli_entry_point()
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        run_public_workflow_primitives(temporary_path)
        normalized_output = run_normalize(temporary_path)
        openvdb_output = run_openvdb(normalized_output, temporary_path)
        acvd_output = run_acvd_cases(temporary_path, openvdb_output)
        run_coacd(acvd_output, temporary_path)
        run_prepare_object(temporary_path)
        run_prepare_objects(temporary_path)
        run_volume_analysis(temporary_path)


if __name__ == "__main__":
    run()
