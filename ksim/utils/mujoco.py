"""Defines utility functions for Mujoco and MJX."""

import mujoco
from jaxtyping import Array
from mujoco import mjx


def update_model_field(model: mujoco.MjModel | mjx.Model, name: str, new_value: Array) -> mujoco.MjModel | mjx.Model:
    """Handles the update of a model field in a mujoco.MjModel or mjx.Model object.

    This is necessary because Mujoco handles model elements as pointers, while
    MJX model elements are stateful.
    """
    if isinstance(model, mujoco.MjModel):
        getattr(model, name)[:] = new_value
    else:
        model = model.replace(**{name: new_value})
    return model


def update_data_field(data: mujoco.MjData | mjx.Data, name: str, new_value: Array) -> mujoco.MjData | mjx.Data:
    """Handles the update of a data field in a mujoco.MjData or mjx.Data object.

    This is necessary because Mujoco handles data elements as pointers, while
    MJX data elements are stateful.

    Args:
        data: The mujoco.MjData or mjx.Data object to update.
        name: The name of the field to update.
        new_value: The new value to set the field to.

    Returns:
        The updated mujoco.MjData or mjx.Data object.
    """
    if isinstance(data, mujoco.MjData):
        getattr(data, name)[:] = new_value
    else:
        data = data.replace(**{name: new_value})
    return data
