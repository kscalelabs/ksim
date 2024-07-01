# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

from ksim.scripts.run_onshape_to_urdf import run_onshape_to_urdf

STOMPY_TORSO_7DOF = (
    "https://cad.onshape.com/documents/77f178833017912bdc0eaaa8/w/"
    "958f1684049f1413f6dd7831/e/348cad65143f892a40e33206"
)

STOMPY_TORSO_5DOF = (
    "https://cad.onshape.com/documents/77f178833017912bdc0eaaa8/"
    "w/958f1684049f1413f6dd7831/e/363614f1fc7c150003de244c"
)


def main() -> None:
    run_onshape_to_urdf(
        model_url=STOMPY_TORSO_7DOF,
        output_dir="stompy_torso_7dof",
        override_central_node="torso_1_hip_mount_1",
    )

    run_onshape_to_urdf(
        model_url=STOMPY_TORSO_5DOF,
        output_dir="stompy_torso_5dof",
        override_central_node="torso_1_hip_mount_1",
    )


if __name__ == "__main__":
    # python -m ksim.scripts.update_stompy_arm_s3
    main()
