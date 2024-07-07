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

UPDATE_NAMES = {
    "dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": "left shoulder pitch",
    "upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": "shoulder pitch",
    "dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": "right shoulder pitch",
    "dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": "left shoulder yaw",
    "upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": "shoulder yaw",
    "dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": "right shoulder yaw",
    "torso_1_rmd_x8_90_mock_1_dof_x8": "torso roll",
    "dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": "left shoulder roll",
    "upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": "shoulder roll",
    "dof_1_full_arm_7_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": "right shoulder roll",
    "dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": "left elbow pitch",
    "upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": "elbow pitch",
    "dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": "right elbow pitch",
    "dof_2_lower_arm_3_dof_1_rmd_x4_24_mock_1_dof_x4": "left wrist roll",
    "lower_arm_3_dof_1_rmd_x4_24_mock_1_dof_x4": "wrist roll",
    "dof_1_lower_arm_3_dof_1_rmd_x4_24_mock_1_dof_x4": "right wrist roll",
    "dof_2_lower_arm_3_dof_1_rmd_x4_24_mock_2_dof_x4": "left wrist pitch",
    "lower_arm_3_dof_1_rmd_x4_24_mock_2_dof_x4": "wrist pitch",
    "dof_1_lower_arm_3_dof_1_rmd_x4_24_mock_2_dof_x4": "right wrist pitch",
    "dof_2_lower_arm_3_dof_1_rmd_x4_24_mock_3_dof_x4": "left wrist yaw",
    "lower_arm_3_dof_1_rmd_x4_24_mock_3_dof_x4": "wrist yaw",
    "dof_1_lower_arm_3_dof_1_rmd_x4_24_mock_3_dof_x4": "right wrist yaw",
    "dof_1_lower_arm_3_dof_1_hand_1_slider_1": "right hand right finger",
    "dof_1_lower_arm_3_dof_1_hand_1_slider_2": "right hand left finger",
    "dof_2_lower_arm_3_dof_1_hand_1_slider_1": "left hand right finger",
    "dof_2_lower_arm_3_dof_1_hand_1_slider_2": "left hand left finger",
    "dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_1_dof_x4": "left wrist roll",
    "lower_arm_1_dof_1_rmd_x4_24_mock_1_dof_x4": "wrist roll",
    "dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_1_dof_x4": "right wrist roll",
    "dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": "left wrist pitch",
    "lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": "wrist pitch",
    "dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": "right wrist pitch",
    "dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_3_dof_x4": "left wrist yaw",
    "lower_arm_1_dof_1_rmd_x4_24_mock_3_dof_x4": "wrist yaw",
    "dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_3_dof_x4": "right wrist yaw",
    "dof_1_lower_arm_1_dof_1_hand_1_slider_1": "right hand right finger",
    "dof_1_lower_arm_1_dof_1_hand_1_slider_2": "right hand  left finger",
    "dof_2_lower_arm_1_dof_1_hand_1_slider_1": "left hand right finger",
    "dof_2_lower_arm_1_dof_1_hand_1_slider_2": "left hand left finger",
    "lower_limbs_1_leg_assembly_2_rmd_x12_150_mock_1_dof_x12": "left hip pitch",
    "lower_limbs_1_leg_assembly_1_rmd_x12_150_mock_1_dof_x12": "right hip pitch",
    "lower_limbs_1_leg_assembly_2_rmd_x8_90_mock_1_dof_x8": "left hip yaw",
    "lower_limbs_1_leg_assembly_1_rmd_x8_90_mock_1_dof_x8": "right hip yaw",
    "lower_limbs_1_leg_assembly_2_rmd_x8_90_mock_2_dof_x8": "left hip roll",
    "lower_limbs_1_leg_assembly_1_rmd_x8_90_mock_2_dof_x8": "right hip roll",
    "lower_limbs_1_leg_assembly_2_rmd_x8_90_mock_3_dof_x8": "left knee pitch",
    "lower_limbs_1_leg_assembly_1_rmd_x8_90_mock_3_dof_x8": "right knee pitch",
    "lower_limbs_1_leg_assembly_2_rmd_x4_24_mock_1_dof_x4": "left ankle pitch",
    "lower_limbs_1_leg_assembly_1_rmd_x4_24_mock_1_dof_x4": "right ankle pitch",
    "lower_limbs_1_left_foot_1_rmd_x4_24_mock_1_dof_x4": "right ankle roll",
    "lower_limbs_1_right_foot_1_rmd_x4_24_mock_1_dof_x4": "left ankle roll",
    "full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": "right shoulder roll",
    "full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": "left shoulder roll",
}

OVERRIDE = [
    "hand_1_rmd_x4_24_mock_1_dof_x4",
    "lower_arm_3_dof_1_hand_1_rmd_x4_24_mock_1_dof_x4",
]

JOINT_LIMITS = {
    "ankle roll": (-2.26893, -1.22173),
    "ankle pitch": (-1.5708, 2.18166),
    "knee pitch": (-4.18879, 0),
    "right hip roll": (0, 3.14159),
    "left hip roll": (-3.14159, 0),
    "right hip yaw": (-1.0472, 2.0944),
    "left hip yaw": (-1.0472, 2.0944),
    "hip pitch": (-4.712389, 4.712389),
    "torso roll": (-4.36332, 4.36332),
    "shoulder pitch": (-4.71239, 4.71239),
    "shoulder yaw": (0.97738438, 5.3058009),
    "shoulder roll": (-4.71239, 4.71239),
    "wrist roll": (-4.71239, 4.71239),
    "left wrist pitch": (-3.66519, -1.39626),
}


def run_onshape_to_urdf(model_url: str, output_dir: str | Path, override_central_node: str | None = None) -> None:
    configure_kol_logging()

    output_dir = Path(output_dir)

    # Gets the latest STL URDF and MJCF.
    converter = Converter(
        document_url=model_url,
        output_dir=output_dir / "latest_robot",
        suffix_to_joint_effort=list(SUFFIX_TO_JOINT_EFFORT.items()),
        disable_mimics=True,
        mesh_ext="stl",
        override_central_node=override_central_node,
        remove_inertia=True,
        merge_fixed_joints=True,
        simplify_meshes=True,
        skip_small_parts=True,
        override_joint_names=UPDATE_NAMES,
        override_nonfixed=OVERRIDE,
        override_limits=JOINT_LIMITS,
    )
    converter.save_mjcf()
    latest_stl_urdf_path = converter.output_dir

    # Manually builds the tarball.
    with tarfile.open(output_dir / "latest_meshes.tar.gz", "w:gz") as tar:
        for suffix in (".urdf", ".stl", ".mjcf"):
            for file in latest_stl_urdf_path.rglob(f"**/*{suffix}"):
                tar.add(file, arcname=file.relative_to(latest_stl_urdf_path))
