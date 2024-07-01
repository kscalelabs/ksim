# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

import tarfile
from pathlib import Path

from kol.logging import configure_logging as configure_kol_logging
from kol.onshape.converter import Converter

SUFFIX_TO_JOINT_EFFORT = {
    "dof_x4_h": 1.5,
    "dof_x4": 1.5,
    "dof_x6": 3,
    "dof_x8": 6,
    "dof_x10": 12,
    "knee_revolute": 13.9,
    "ankle_revolute": 6,
}


def run_onshape_to_urdf(model_url: str, output_dir: str | Path, override_central_node: str | None = None) -> None:
    configure_kol_logging()

    output_dir = Path(output_dir)

    # Gets the latest STL URDF.
    converter = Converter(
        document_url=model_url,
        output_dir=output_dir / "latest_stl_urdf",
        suffix_to_joint_effort=list(SUFFIX_TO_JOINT_EFFORT.items()),
        disable_mimics=True,
        mesh_ext="stl",
        override_central_node=override_central_node,
    )
    converter.save_urdf()
    latest_stl_urdf_path = converter.output_dir

    # Manually builds the tarball.
    with tarfile.open(output_dir / "latest_stl_urdf.tar.gz", "w:gz") as tar:
        for suffix in (".urdf", ".stl"):
            for file in latest_stl_urdf_path.rglob(f"**/*{suffix}"):
                tar.add(file, arcname=file.relative_to(latest_stl_urdf_path))

    # Gets the latest OBJ URDF.
    converter = Converter(
        document_url=model_url,
        output_dir=output_dir / "latest_obj_urdf",
        suffix_to_joint_effort=list(SUFFIX_TO_JOINT_EFFORT.items()),
        disable_mimics=True,
        mesh_ext="obj",
        override_central_node=override_central_node,
    )
    converter.save_urdf()
    latest_obj_urdf_path = converter.output_dir

    # Manually builds the tarball.
    with tarfile.open(output_dir / "latest_obj_urdf.tar.gz", "w:gz") as tar:
        for suffix in (".urdf", ".obj"):
            for file in latest_obj_urdf_path.rglob(f"**/*{suffix}"):
                tar.add(file, arcname=file.relative_to(latest_obj_urdf_path))

    # Gets the latest MJCF.
    converter = Converter(
        document_url=model_url,
        output_dir=output_dir / "latest_mjcf",
        suffix_to_joint_effort=list(SUFFIX_TO_JOINT_EFFORT.items()),
        disable_mimics=True,
        override_central_node=override_central_node,
    )
    converter.save_mjcf()
    latest_mjcf_path = converter.output_dir

    # Manually builds the tarball.
    with tarfile.open(output_dir / "latest_mjcf.tar.gz", "w:gz") as tar:
        for suffix in (".xml", ".stl"):
            for file in latest_mjcf_path.rglob(f"**/*{suffix}"):
                tar.add(file, arcname=file.relative_to(latest_mjcf_path))
