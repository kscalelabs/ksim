# mypy: disable-error-code="import-not-found"
"""Updates the Stompy model."""

from ksim.scripts.run_onshape_to_urdf import run_onshape_to_urdf

STOMPY_MODEL = (
    "https://cad.onshape.com/documents/71f793a23ab7562fb9dec82d/"
    "w/6160a4f44eb6113d3fa116cd/e/1a95e260677a2d2d5a3b1eb3"
)


def main() -> None:
    run_onshape_to_urdf(
        model_url=STOMPY_MODEL,
        output_dir="stompy",
    )


if __name__ == "__main__":
    # python -m ksim.scripts.update_stompy_s3
    main()
