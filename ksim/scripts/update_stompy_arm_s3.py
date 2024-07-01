# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

from ksim.scripts.run_onshape_to_urdf import run_onshape_to_urdf

STOMPY_ARM_MODEL = (
    "https://cad.onshape.com/documents/afaee604f6ca311526a6aec8/"
    "w/29af84cb974c2d825b71de39/e/4fef6bce7179a665e62b03ba"
)


def main() -> None:
    run_onshape_to_urdf(
        model_url=STOMPY_ARM_MODEL,
        output_dir="stompy_arm",
        override_central_node="left_arm_1_arm_part_1_1",
    )


if __name__ == "__main__":
    # python -m ksim.scripts.update_stompy_arm_s3
    main()
