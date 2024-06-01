# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

from ksim.scripts.run_onshape_to_urdf import run_onshape_to_urdf

STOMPY_ARM_MODEL = (
    "https://cad.onshape.com/documents/c18f0f88a92c5eb7fe8968b1/"
    "w/db6d2ff05955edd31f39fda3/e/b6b0d3abb4be86fb66d3b701"
)


def main() -> None:
    run_onshape_to_urdf(
        model_url=STOMPY_ARM_MODEL,
        output_dir="stompy_arm",
        override_central_node="arm_part_1_1",
    )


if __name__ == "__main__":
    # python -m ksim.scripts.update_stompy_arm_s3
    main()
