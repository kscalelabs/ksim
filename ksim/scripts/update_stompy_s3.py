# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

from ksim.scripts.run_onshape_to_urdf import run_onshape_to_urdf

STOMPY_MODEL_7DOF = (
    "https://cad.onshape.com/documents/71f793a23ab7562fb9dec82d/"
    "w/e879bcf272425973f9b3d8ad/e/1a95e260677a2d2d5a3b1eb3"
)

STOMPY_MODEL_5DOF = (
    "https://cad.onshape.com/documents/71f793a23ab7562fb9dec82d/"
    "w/e879bcf272425973f9b3d8ad/e/e07509571989528061c06a08"
)


def main() -> None:
    run_onshape_to_urdf(
        model_url=STOMPY_MODEL_7DOF,
        output_dir="stompy_7dof",
    )

    run_onshape_to_urdf(
        model_url=STOMPY_MODEL_5DOF,
        output_dir="stompy_5dof",
    )


if __name__ == "__main__":
    # python -m ksim.scripts.update_stompy_s3
    main()
